"""Length control: truncate each unsteered output to the word count of the steered output for
the same prompt, then score it like any other condition.

usage: python src/trunc_control.py build     -> results/gens/trunc_ctrl.jsonl
       (then: python src/score.py results/gens/trunc_ctrl.jsonl; python src/judge.py results/gens/trunc_ctrl.jsonl)
       python src/trunc_control.py table     -> results/truncation_control.csv
"""
import sys

import pandas as pd

CONDS = ["ai_read_c-0.5", "ai_read_c-0.75", "write_chat_c-0.5", "chat_register_c-0.75", "ai_read_ablate"]
DESKLIB_THR = 0.725  # 5% FPR threshold on test human references (results/analysis_main_test.json)

if sys.argv[1] == "build":
    d = pd.read_json("results/gens/main_test.jsonl", lines=True)
    none = d[d.cond == "none"].set_index("pid").text
    rows = []
    for c in CONDS:
        s = d[d.cond == c].set_index("pid").text
        for pid in none.index:
            n = max(len(s[pid].split()), 1)
            rows.append(dict(cond="trunc_like_" + c, pid=pid, text=" ".join(none[pid].split()[:n])))
    pd.DataFrame(rows).to_json("results/gens/trunc_ctrl.jsonl", orient="records", lines=True)
else:
    t = pd.read_parquet("results/scores/trunc_ctrl.parquet")
    m = pd.read_parquet("results/scores/main_test.parquet")
    t = t.merge(pd.read_parquet("results/judge/trunc_ctrl.parquet")[["cond", "pid", "ai_likelihood"]], on=["cond", "pid"], how="left")
    m = m.merge(pd.read_parquet("results/judge/main_test.parquet")[["cond", "pid", "ai_likelihood"]], on=["cond", "pid"])
    none = m[m.cond == "none"].set_index("pid")
    rows = []
    for c in t.cond.unique():
        a, s = t[t.cond == c].set_index("pid"), m[m.cond == c.replace("trunc_like_", "")].set_index("pid")
        r = dict(cond=c)
        for k in ["desklib", "hc3rob", "bino", "radar", "ai_likelihood", "nwords"]:
            r[k + "_none"], r[k + "_trunc"], r[k + "_steer"] = none[k].mean(), a[k].mean(), s[k].mean()
        ok = a.ai_likelihood.dropna().index          # judge scores exist only for a subset (API limit)
        r["n_judged"] = len(ok)
        r["judge_trunc_sub"], r["judge_steer_sub"] = a.loc[ok, "ai_likelihood"].mean(), s.loc[ok, "ai_likelihood"].mean()
        r["judge_none_sub"] = none.loc[ok, "ai_likelihood"].mean()
        r["desklib_flag_trunc"], r["desklib_flag_steer"] = (a.desklib > DESKLIB_THR).mean(), (s.desklib > DESKLIB_THR).mean()
        rows.append(r)
    R = pd.DataFrame(rows).set_index("cond").T.round(3)
    R.to_csv("results/truncation_control.csv")
    print(R.to_string())
