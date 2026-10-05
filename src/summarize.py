"""Final summary tables for the report: key conditions, Holm-corrected tests against the
equal-norm random controls, per-task breakdown, readback, judge/detector agreement.

usage: python src/summarize.py  -> results/summary.json (+ printed tables)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
DETS = ["desklib", "bino", "radar", "hc3rob", "ai_likelihood"]
out = {}


def load(run):
    S = pd.read_parquet(f"results/scores/{run}.parquet")
    J = pd.read_parquet(f"results/judge/{run}.parquet")
    S = S.merge(J[["cond", "pid", "ai_likelihood", "coherence", "on_task"]], on=["cond", "pid"], how="left")
    try:
        R = pd.read_parquet(f"results/scores/{run}_readback.parquet")
        S = S.merge(R[["cond", "pid", "readback", "self_nll"]], on=["cond", "pid"], how="left")
    except FileNotFoundError:
        pass
    return S


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for i, o in enumerate(order):
        run = max(run, (len(ps) - i) * ps[o])
        adj[o] = min(run, 1.0)
    return adj


def boot(x, n=2000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


for run, rands in [("main_test", ["random0", "random1"]), ("base_cont", ["random0", "random1"]),
                   ("instruct_cont", ["random0"])]:
    S = load(run)
    conds = list(dict.fromkeys(S.cond))
    W = {m: S.pivot(index="pid", columns="cond", values=m) for m in DETS + ["coherence", "on_task", "readback"] if m in S}
    # --- every steered condition vs the mean of the random controls at the same signed coefficient
    rows = []
    for c in conds:
        if "_c" not in c or c.split("_c")[0] in rands:
            continue
        coef = c.split("_c")[1]
        rc = [f"{r}_c{coef}" for r in rands if f"{r}_c{coef}" in conds]
        if not rc:
            continue
        for m in DETS + ["coherence", "on_task"]:
            d = (W[m][c] - W[m][rc].mean(axis=1)).dropna()
            if len(d) < 20 or (d == 0).all():
                continue
            lo, hi = boot(d)
            rows.append(dict(cond=c, metric=m, diff_vs_random=d.mean(), ci_lo=lo, ci_hi=hi, n=len(d),
                             p=wilcoxon(d).pvalue))
    V = pd.DataFrame(rows)
    V["p_holm"] = np.nan
    for m, g in V.groupby("metric"):          # Holm within each metric across all conditions
        V.loc[g.index, "p_holm"] = holm(g.p.values)
    V.to_csv(f"results/vs_random_{run}.csv", index=False)
    piv = V.pivot(index="cond", columns="metric", values="diff_vs_random").reindex([c for c in conds if c in set(V.cond)])
    sig = V.pivot(index="cond", columns="metric", values="p_holm").reindex(piv.index)
    print(f"\n=== {run}: difference vs equal-norm random controls (paired mean; * = Holm p < 0.05)")
    print((piv.round(3).astype(str) + sig.map(lambda p: "*" if p < 0.05 else "")).to_string())
    if "readback" in W:
        rb = W["readback"].mean().reindex(conds)
        print("\nreadback (probe projection of the generated text, human-sd units):")
        print(rb.round(2).to_string())
        out[f"{run}_readback"] = rb.round(3).to_dict()

# ---------------------------------------------------------------- main run extras
S = load("main_test")
P = {p["id"]: p for p in json.load(open("results/data/prompts.json"))["test"]}
S["task"] = S.pid.map(lambda i: P[i]["task"])
key = ["none", "prompt_human", "ai_read_c-0.5", "ai_read_c-0.75", "write_chat_c-0.5", "chat_register_c-0.5",
       "random0_c-0.5", "ai_read_ablate", "ai_read_c+0.5"]
bt = S[S.cond.isin(key)].groupby(["task", "cond"])[["desklib", "bino", "ai_likelihood", "coherence", "on_task"]].mean()
print("\n=== per-task breakdown (continue = HAP-E, answer = HC3, write = RAID)")
print(bt.round(3).unstack(0).reindex(key).to_string())
bt.round(4).reset_index().to_csv("results/per_task_main_test.csv", index=False)

# agreement between independent measures across all generated texts
ag = S[DETS + ["readback"]].corr(method="spearman") if "readback" in S else S[DETS].corr(method="spearman")
print("\n=== Spearman correlation between measures over all generated texts")
print(ag.round(2).to_string())
out["measure_agreement"] = ag.round(3).to_dict()

# noise floor: two unsteered samples with different seeds
n1, n2 = S[S.cond == "none"].set_index("pid"), S[S.cond == "none_seed2"].set_index("pid")
out["seed_noise"] = {m: float((n2[m] - n1[m]).mean()) for m in DETS + ["coherence", "on_task"]}
print("\nunsteered seed-to-seed mean difference:", {k: round(v, 3) for k, v in out["seed_noise"].items()})
json.dump(out, open("results/summary.json", "w"), indent=1)
