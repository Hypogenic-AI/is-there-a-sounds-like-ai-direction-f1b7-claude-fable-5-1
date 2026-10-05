"""Generate text under residual-stream interventions.

usage: python src/steer.py --model instruct --split test --plan main --out main_test
A condition is (name, kind, direction, layer, coef):
  kind 'add'    : add coef * mean_resid_norm[layer] * unit(direction[layer]) at hidden state `layer`,
                  every position (prompt and generated tokens). coef < 0 moves toward the human side
                  for AI directions.
  kind 'ablate' : project unit(direction[layer]) out of every block's output.
  kind 'prompt' : no intervention, system prompt given in `direction`.
  kind 'none'   : unsteered.
Appends to results/gens/<out>.jsonl; finished conditions are skipped on restart.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from common import generate, load_model, load_prompts, make_contexts, steering, unit

HUMAN_PROMPT = ("Write like a human, not like an AI. Your text must read as if a real person wrote it: "
                "natural and unpolished where appropriate, with none of the phrasing, structure or tone "
                "typical of AI assistants.")
AI_PROMPT = ("Write in the unmistakable style of an AI assistant: polished, balanced, well-structured, "
             "with the phrasing and tone typical of AI-generated text.")
CONFOUNDS = ["formality", "fluency", "length", "chat_register", "assistant_axis"]


def get_directions(model_key, dir_model=None):
    """Unit directions per layer, including derived ones (random, residualised)."""
    D = dict(np.load(f"results/directions_{dir_model or model_key}.npz"))
    R = json.load(open(f"results/readout_{dir_model or model_key}.json"))
    norms = {int(l): r["resid_norm"] for l, r in R["layers"].items()}
    gaps = {int(l): r["dm_norm"] for l, r in R["layers"].items()}
    U = {}
    nl, d = D["ai_read"].shape
    for k, v in D.items():
        if k == "genre_basis":
            continue
        U[k] = np.stack([np.zeros(d)] + [unit(v[l]) for l in range(1, nl)])
    rng = np.random.default_rng(123)
    for s in range(3):
        r = unit(rng.standard_normal(d))
        U[f"random{s}"] = np.tile(r, (nl, 1))
    # AI direction with the span of all confound directions (+ genre subspace) removed
    perp, perp_form, perp_axis = [np.zeros(d)], [np.zeros(d)], [np.zeros(d)]
    for l in range(1, nl):
        def rm(names, genre=False):
            B = [D[c][l] for c in names if c in D] + (list(D["genre_basis"][l]) if genre else [])
            Q, _ = np.linalg.qr(np.asarray(B, np.float64).T)
            u = U["ai_read"][l]
            return unit(u - Q @ (Q.T @ u))
        perp.append(rm(CONFOUNDS, True)); perp_form.append(rm(["formality"]))
        perp_axis.append(rm(["assistant_axis", "chat_register"]))
    U["ai_read_perp_all"], U["ai_read_perp_formality"], U["ai_read_perp_persona"] = map(np.stack, (perp, perp_form, perp_axis))
    return U, norms, gaps


def plan_pilot():
    conds = [("none", "none", None, 0, 0.0)]
    for l in (8, 12, 16, 20):
        for c in (-0.25, -0.5, -1.0):
            conds.append((f"ai_read_L{l}_c{c}", "add", "ai_read", l, c))
            conds.append((f"random0_L{l}_c{c}", "add", "random0", l, c))
    return conds


def plan_main(L, coefs):
    """coefs: positive magnitudes (fractions of the mean residual norm at layer L).
    Conditions are ordered by priority; second-tier directions use the two middle coefficients."""
    conds = [("none", "none", None, 0, 0.0), ("none_seed2", "none", None, 0, 0.0),
             ("prompt_human", "prompt", HUMAN_PROMPT, 0, 0.0), ("prompt_ai", "prompt", AI_PROMPT, 0, 0.0)]
    neg = lambda names, cs: [(f"{n}_c-{c}", "add", n, L, -c) for n in names for c in cs]
    pos = lambda names, cs: [(f"{n}_c+{c}", "add", n, L, c) for n in names for c in cs]
    conds += neg(["ai_read", "random0", "write_chat", "chat_register", "formality", "assistant_axis"], coefs)
    conds += pos(["ai_read", "random0"], coefs[:3])
    conds += neg(["ai_read_perp_all"], coefs)
    conds += [(f"{n}_ablate", "ablate", n, L, 0.0) for n in ["ai_read", "write_chat", "random0", "formality"]]
    conds += neg(["random1"], coefs)
    mid = coefs[1:3]
    conds += neg(["ai_lr", "fluency", "length", "ai_read_perp_persona", "ai_base_gens", "ai_hc3"], mid)
    conds += pos(["write_chat", "chat_register", "ai_lr"], coefs[:2])
    return conds


def plan_base(L, coefs):
    """Raw continuation: push toward the AI side (+) and the human side (-)."""
    conds = [("none", "none", None, 0, 0.0)]
    for name in ["ai_read", "write_raw", "chat_register", "random0", "random1"]:
        for c in coefs:
            conds.append((f"{name}_c+{c}", "add", name, L, c))
            conds.append((f"{name}_c-{c}", "add", name, L, -c))
    return conds


def plan_rawsmall(L, coefs):
    conds = [("none", "none", None, 0, 0.0)]
    for name in ["ai_read", "random0"]:
        for c in coefs:
            conds.append((f"{name}_c+{c}", "add", name, L, c))
            conds.append((f"{name}_c-{c}", "add", name, L, -c))
    return conds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="instruct")
    ap.add_argument("--dir-model", default=None, help="model whose directions are used (default: same)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--plan", default="main")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--coefs", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=104)
    a = ap.parse_args()
    coefs = [float(c) for c in a.coefs.split(",")]
    conds = {"pilot": plan_pilot, "main": lambda: plan_main(a.layer, coefs),
             "base": lambda: plan_base(a.layer, coefs), "rawsmall": lambda: plan_rawsmall(a.layer, coefs)}[a.plan]()
    U, norms, gaps = get_directions(a.model, a.dir_model)
    prompts = load_prompts()[a.split]
    tok, model = load_model(a.model)
    out = f"results/gens/{a.out}.jsonl"
    done = set(pd.read_json(out, lines=True).cond) if os.path.exists(out) else set()
    json.dump([dict(name=n, kind=k, direction=d if k != "prompt" else "system prompt", layer=l, coef=c,
                    abs_norm=abs(c) * norms.get(l, 0), dm_gap=gaps.get(l, 0)) for n, k, d, l, c in conds],
              open(f"results/gens/{a.out}_conds.json", "w"), indent=1)
    for name, kind, dname, l, c in conds:
        if name in done:
            continue
        if kind == "add" and dname not in U:
            print("skip (no direction)", name); continue
        ctx = make_contexts(tok, prompts, system=dname if kind == "prompt" else None)
        add = [(l, torch.tensor(c * norms[l] * U[dname][l]))] if kind == "add" else None
        abl = torch.tensor(U[dname][l]) if kind == "ablate" else None
        with steering(model, add=add, ablate=abl):
            gens = generate(tok, model, ctx, seed=2 if name == "none_seed2" else 0, bs=a.bs)
        pd.DataFrame(dict(cond=name, pid=[p["id"] for p in prompts], text=gens)).to_json(
            out, orient="records", lines=True, mode="a")
        print(name, "|", gens[0][:150].replace("\n", " "), flush=True)
