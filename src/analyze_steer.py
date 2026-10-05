"""Analysis of steering runs: detector validation, per-condition table, paired statistics,
matched-coherence comparison.

usage: python src/analyze_steer.py <run name> <prompt split> [--nojudge]
Reads results/scores/<run>.parquet (+ _emb.npy), results/scores/refs.parquet,
results/judge/<run>.parquet (optional). Writes results/analysis_<run>.json and
results/table_<run>.csv.
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

run, split = sys.argv[1], sys.argv[2]
DETS = ["desklib", "radar", "hc3rob", "bino", "ai_likelihood"]
rng = np.random.default_rng(0)

S = pd.read_parquet(f"results/scores/{run}.parquet")
emb = np.load(f"results/scores/{run}_emb.npy")
S["row"] = np.arange(len(S))
try:
    J = pd.read_parquet(f"results/judge/{run}.parquet")
    S = S.merge(J[["cond", "pid", "ai_likelihood", "coherence", "on_task"]], on=["cond", "pid"], how="left")
except FileNotFoundError:
    for k in ("ai_likelihood", "coherence", "on_task"):
        S[k] = np.nan
refs = pd.read_parquet("results/scores/refs.parquet").query("split == @split")
try:
    Jr = pd.read_parquet("results/judge/refs.parquet")
    refs = refs.merge(Jr[["pid", "ai_likelihood", "coherence", "on_task"]].drop_duplicates("pid"), on="pid", how="left")
except FileNotFoundError:
    refs["ai_likelihood"] = np.nan; refs["coherence"] = np.nan; refs["on_task"] = np.nan

none = S[S.cond == "none"].set_index("pid")
pids = list(none.index)

# ---------------------------------------------------------------- detector validation
val = {}
thr = {}
for d in DETS:
    a, h = none[d].dropna(), refs[d].dropna()
    if len(a) < 10 or len(h) < 10:
        continue
    thr[d] = float(np.quantile(h, 0.95))                      # 5% FPR on matched human references
    val[d] = dict(auroc=float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(h))], np.r_[a, h])),
                  tpr_at_5fpr=float((a > thr[d]).mean()), thr=thr[d],
                  mean_unsteered=float(a.mean()), mean_human=float(h.mean()))
print("detector validation (unsteered vs human refs):")
print(pd.DataFrame(val).T.round(3))

# ---------------------------------------------------------------- per-condition table
METRICS = DETS + ["coherence", "on_task", "ext_nll", "distinct3", "formality", "nwords", "sim_none"]


def boot_ci(x, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 5:
        return [np.nan, np.nan]
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


rows, per = [], {}
none_emb = emb[none.loc[pids, "row"].values]
for cond, g in S.groupby("cond", sort=False):
    g = g.set_index("pid").reindex(pids)
    g["sim_none"] = (emb[g.row.values.astype(int)] * none_emb).sum(1)
    per[cond] = g
    r = dict(cond=cond, n=len(g))
    for m in METRICS:
        r[m] = g[m].mean()
    for d in thr:
        r[f"{d}_flag"] = (g[d] > thr[d]).mean() if g[d].notna().any() else np.nan
        dlt = (g[d] - none[d].reindex(pids)).dropna()
        r[f"{d}_delta"] = dlt.mean()
        r[f"{d}_ci_lo"], r[f"{d}_ci_hi"] = boot_ci(dlt)
        try:
            r[f"{d}_p"] = wilcoxon(dlt).pvalue if cond != "none" and (dlt != 0).any() else np.nan
        except ValueError:
            r[f"{d}_p"] = np.nan
    for m in ("coherence", "on_task"):
        dlt = (g[m] - none[m].reindex(pids)).dropna()
        r[f"{m}_delta"] = dlt.mean() if len(dlt) else np.nan
    rows.append(r)
hr = dict(cond="human_ref", n=len(refs))
for m in METRICS:
    hr[m] = refs[m].mean() if m in refs else np.nan
for d in thr:
    hr[f"{d}_flag"] = (refs[d] > thr[d]).mean()
rows.append(hr)
T = pd.DataFrame(rows)
m = T.cond.str.extract(r"^(.*)_c([+-][\d.]+)$")
T["direction"], T["coef"] = m[0], m[1].astype(float)
T.to_csv(f"results/table_{run}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
show = ["cond", "desklib", "radar", "bino", "ai_likelihood", "desklib_flag", "bino_flag", "coherence", "on_task",
        "ext_nll", "distinct3", "formality", "nwords", "sim_none"]
print(T[[c for c in show if c in T]].round(3).to_string())

# ---------------------------------------------------------------- AI direction vs random at equal norm (paired)
out = dict(validation=val, vs_random={}, matched_coherence={})
rand_names = sorted({d for d in T.direction.dropna() if re.fullmatch(r"random\d", d)})
for _, r in T.dropna(subset=["direction"]).iterrows():
    if r.direction in rand_names:
        continue
    rc = [f"{n}_c{r.coef:+g}".replace("+-", "-") for n in rand_names]
    rc = [c for c in [f"{n}_c{'+' if r.coef > 0 else '-'}{abs(r.coef)}" for n in rand_names] if c in per]
    if not rc:
        continue
    res = {}
    for d in list(thr) + ["coherence", "on_task"]:
        rmean = np.mean([per[c][d].values.astype(float) for c in rc], 0)
        dlt = per[r.cond][d].values.astype(float) - rmean
        dlt = dlt[~np.isnan(dlt)]
        if len(dlt) < 10:
            continue
        try:
            p = float(wilcoxon(dlt).pvalue)
        except ValueError:
            p = np.nan
        res[d] = dict(diff=float(dlt.mean()), ci=boot_ci(dlt), p=p)
    out["vs_random"][r.cond] = res

# ---------------------------------------------------------------- matched coherence
# For each direction and sign: the largest |coef| whose mean judge coherence and on-task score are
# both within 10 points of unsteered. Report detector deltas there.
if T.coherence.notna().any():
    c0, t0 = none.coherence.mean(), none.on_task.mean()
    for (dname, sign), g in T.dropna(subset=["direction"]).assign(sign=lambda x: np.sign(x.coef)).groupby(["direction", "sign"]):
        ok = g[(g.coherence >= c0 - 10) & (g.on_task >= t0 - 10)]
        if len(ok) == 0:
            out["matched_coherence"][f"{dname}{'+' if sign > 0 else '-'}"] = None
            continue
        b = ok.loc[ok.coef.abs().idxmax()]
        out["matched_coherence"][f"{dname}{'+' if sign > 0 else '-'}"] = {
            "coef": float(b.coef), "coherence": float(b.coherence), "on_task": float(b.on_task),
            **{f"{d}_delta": float(b[f"{d}_delta"]) for d in thr},
            **{f"{d}_ci": [float(b[f"{d}_ci_lo"]), float(b[f"{d}_ci_hi"])] for d in thr},
            **{f"{d}_flag": float(b[f"{d}_flag"]) for d in thr}, "sim_none": float(b.sim_none)}
    print("\nmatched coherence (largest |coef| with coherence and on_task within 10 points of unsteered):")
    print(pd.DataFrame({k: v for k, v in out["matched_coherence"].items() if v}).T.round(3).to_string())
json.dump(out, open(f"results/analysis_{run}.json", "w"), indent=1)
