"""Activations for 'writing-mode' directions and the assistant-axis proxy.

usage: python src/fit_acts.py instruct|base

(a) chat fit prompts (instruct only): the model's own unsteered response vs the matched
    human reference teacher-forced in the assistant slot of the same prompt.
(b) raw continuation prompts (both models): own continuation vs the true human continuation
    after the same raw prefix (no chat template).
(c) assistant-axis proxy (instruct only): default-assistant responses vs role-play responses
    (reduced version of Lu et al.: 40 roles x 8 questions, no judge filtering).
Saves results/acts/{model}_fit.npz and the generated texts in results/gens/.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from common import ACTS, chat_prefix, generate, load_model, load_prompts, make_contexts, pooled_acts

key = sys.argv[1]
os.makedirs("results/gens", exist_ok=True)
tok, model = load_model(key)
P = load_prompts()
save = {}


def self_vs_human(prompts, tag):
    ctx = make_contexts(tok, prompts)
    gens = generate(tok, model, ctx, seed=1)
    pd.DataFrame(dict(cond=f"{key}_unsteered", pid=[p["id"] for p in prompts], text=gens)).to_json(
        f"results/gens/fit_{tag}_{key}.jsonl", orient="records", lines=True)
    keep = [i for i, g in enumerate(gens) if len(g.split()) >= 30]
    a_self, nll_s, _ = pooled_acts(tok, model, [ctx[i] for i in keep], [gens[i] for i in keep])
    a_hum, nll_h, _ = pooled_acts(tok, model, [ctx[i] for i in keep], [prompts[i]["ref"] for i in keep])
    save[f"{tag}_self"], save[f"{tag}_human"] = a_self, a_hum
    save[f"{tag}_task"] = np.array([prompts[i]["task"] for i in keep])
    print(tag, "kept", len(keep), "self nll", nll_s.mean(), "human nll", nll_h.mean())


self_vs_human(P["cont_fit"], "rawcont")
if key == "instruct":
    self_vs_human(P["fit"], "chat")

    # ---- assistant-axis proxy
    rng = np.random.default_rng(0)
    qs = [json.loads(l)["question"] for l in open("code/assistant-axis/data/extraction_questions.jsonl")]
    qs = list(rng.permutation(qs)[:40])
    role_files = sorted(glob.glob("code/assistant-axis/data/roles/instructions/*.json"))
    role_files = [f for f in role_files if "default" not in f and "assistant" not in os.path.basename(f)]
    role_files = list(rng.permutation(role_files)[:40])
    ctx, grp = [], []
    for sysmsg in [None, "You are a helpful assistant.", "You are an AI assistant.",
                   "You are a large language model."]:
        for q in qs:
            ctx.append(chat_prefix(tok, q, sysmsg)); grp.append("default")
    for f in role_files:
        inst = json.load(open(f))["instruction"][0]["pos"]
        for q in rng.permutation(qs)[:8]:
            ctx.append(chat_prefix(tok, q, inst)); grp.append(os.path.basename(f)[:-5])
    gens = generate(tok, model, ctx, seed=2, max_new_tokens=120)
    acts, _, _ = pooled_acts(tok, model, ctx, [g if g else "." for g in gens])
    grp = np.array(grp)
    save["axis_default"] = acts[grp == "default"].astype(np.float32).mean(0)
    save["axis_roles"] = np.stack([acts[grp == r].astype(np.float32).mean(0) for r in sorted(set(grp) - {"default"})])
    pd.DataFrame(dict(cond=grp, pid=np.arange(len(gens)), text=gens)).to_json(
        "results/gens/assistant_axis_proxy.jsonl", orient="records", lines=True)

np.savez(f"{ACTS}/{key}_fit.npz", **save)
print("saved", {k: v.shape for k, v in save.items()})
