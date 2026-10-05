"""E3/E4 geometry that needs raw activations: disjoint-data cosines, base-vs-instruct comparison,
own-writing projections, fluency/genre partialling, Biber-feature correlates.

usage: python src/geometry.py   -> results/geometry.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ACTS, DATA, unit

corpus = pd.read_parquet(f"{DATA}/corpus.parquet")
LAYERS = [4, 8, 12, 16, 20, 24, 28]
A = {k: np.load(f"{ACTS}/{k}_corpus.npy", mmap_mode="r") for k in ("instruct", "base")}
FIT = {k: np.load(f"{ACTS}/{k}_fit.npz") for k in ("instruct", "base")}
DIR = {k: np.load(f"results/directions_{k}.npz") for k in ("instruct", "base")}
META = {k: pd.read_parquet(f"{ACTS}/{k}_corpus_meta.parquet") for k in ("instruct", "base")}
out = {}


def idx(q):
    return np.sort(corpus.query(q).index.values)


def X(k, l, q):
    return np.asarray(A[k][idx(q), l], np.float32)


def cos(a, b):
    return float(unit(a) @ unit(b))


def auc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


TR, TE = "dataset=='hape' and split=='train'", "dataset=='hape' and split=='test'"
GPT = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]
LI = ["llama-3-8B-Instruct", "llama-3-70B-Instruct"]
LB = ["llama-3-8B", "llama-3-70B"]

for l in LAYERS:
    r = {}
    for k in ("instruct", "base"):
        h = X(k, l, f"{TR} and source=='human'")
        v_gpt = X(k, l, f"{TR} and source in @GPT").mean(0) - h.mean(0)          # human -> GPT-4o text
        v_li = X(k, l, f"{TR} and source in @LI").mean(0) - h.mean(0)            # human -> Llama-instruct text
        v_lb = X(k, l, f"{TR} and source in @LB").mean(0) - h.mean(0)            # human -> Llama-base text
        # chat register from the 70B pair only, AI direction from GPT only: no shared texts
        reg8 = X(k, l, f"{TR} and source=='llama-3-8B-Instruct'").mean(0) - X(k, l, f"{TR} and source=='llama-3-8B'").mean(0)
        reg70 = X(k, l, f"{TR} and source=='llama-3-70B-Instruct'").mean(0) - X(k, l, f"{TR} and source=='llama-3-70B'").mean(0)
        u = unit(DIR[k]["ai_read"][l])
        th = X(k, l, f"{TE} and source=='human'")
        mu, sd = (th @ u).mean(), (th @ u).std()
        d = dict(
            cos_gptdir_vs_register8=cos(v_gpt, reg8), cos_gptdir_vs_register70=cos(v_gpt, reg70),
            cos_register8_vs_register70=cos(reg8, reg70),
            cos_gptdir_vs_llamainstdir=cos(v_gpt, v_li), cos_gptdir_vs_llamabasedir=cos(v_gpt, v_lb),
            norm_human_to_gpt=float(np.linalg.norm(v_gpt)), norm_human_to_llamainst=float(np.linalg.norm(v_li)),
            norm_human_to_llamabase=float(np.linalg.norm(v_lb)),
            # how much of the human->base-text shift lies along the AI direction
            frac_basedir_along_ai=float((unit(v_lb) @ u) ** 2),
        )
        # GPT-only direction: transfer to Llama text of both kinds (no generator overlap)
        ug = unit(v_gpt)
        d["gptdir_auc_llama_instruct"] = auc(X(k, l, f"{TE} and source in @LI") @ ug, th @ ug)
        d["gptdir_auc_llama_base"] = auc(X(k, l, f"{TE} and source in @LB") @ ug, th @ ug)
        # own writing: projection of the model's own continuation vs the human continuation
        # (same raw prefix), in units of the human-text sd along the direction
        f = FIT[k]
        ps, ph = f["rawcont_self"][:, l].astype(np.float32) @ u, f["rawcont_human"][:, l].astype(np.float32) @ u
        d["own_rawcont_dprime"] = float((ps.mean() - ph.mean()) / sd)
        d["own_rawcont_auc"] = auc(ps, ph)
        if k == "instruct":
            ps, ph = f["chat_self"][:, l].astype(np.float32) @ u, f["chat_human"][:, l].astype(np.float32) @ u
            d["own_chat_dprime"] = float((ps.mean() - ph.mean()) / sd)
            d["own_chat_auc"] = auc(ps, ph)
        # fluency: NLL alone as a detector, and AUROC of the projection after regressing out NLL + genre
        te = corpus.loc[idx(f"{TE} and (source=='human' or source in @GPT or source in @LI)")]
        y = te.label.values
        nll = META[k].nll.values[te.index.values]
        proj = np.asarray(A[k][te.index.values, l], np.float32) @ u
        Z = np.c_[np.ones(len(te)), nll, pd.get_dummies(te.domain).values[:, 1:].astype(float)]
        resid = proj - Z @ np.linalg.lstsq(Z, proj, rcond=None)[0]
        d["auc_neg_nll_alone"] = float(roc_auc_score(y, -nll))
        d["auc_proj"] = float(roc_auc_score(y, proj))
        d["auc_proj_resid_nll_genre"] = float(roc_auc_score(y, resid))
        d["corr_proj_nll_within_human"] = float(np.corrcoef(proj[y == 0], nll[y == 0])[0, 1])
        d["corr_proj_nll_within_ai"] = float(np.corrcoef(proj[y == 1], nll[y == 1])[0, 1])
        r[k] = d
    # base vs instruct model, same texts
    ub, ui = unit(DIR["base"]["ai_read"][l]), unit(DIR["instruct"]["ai_read"][l])
    r["cos_ai_read_base_vs_instruct"] = float(ub @ ui)
    ix = idx(f"{TR}")
    shift = np.asarray(A["instruct"][ix, l], np.float32).mean(0) - np.asarray(A["base"][ix, l], np.float32).mean(0)
    r["cos_model_shift_vs_ai_read_instruct"] = cos(shift, ui)
    r["cos_model_shift_vs_ai_read_base"] = cos(shift, ub)
    r["model_shift_norm"] = float(np.linalg.norm(shift))
    # cross-model direction transfer: base-model direction applied to instruct activations
    r["auc_basedir_on_instruct_acts"] = auc(X("instruct", l, f"{TE} and (source in @GPT or source in @LI)") @ ub,
                                            X("instruct", l, f"{TE} and source=='human'") @ ub)
    # own writing of both models measured with the *same* (base-model) direction is not possible
    # across models' activations, so compare d' values above instead.
    out[l] = r
    print(l, json.dumps(r)[:900], flush=True)

# ---- Biber-feature correlates of the projection (layer 12, instruct model, HAP-E test)
l = 12
u = unit(DIR["instruct"]["ai_read"][l])
te = corpus.loc[idx(f"{TE}")]
proj = pd.Series(np.asarray(A["instruct"][te.index.values, l], np.float32) @ u, index=te.index)
frames = []
name = {"human": "human-chunk-2"}
for s in te.source.unique():
    b = pd.read_parquet(f"datasets/human_ai_parallel_corpus_biber/biber_data/hape-biber_{name.get(s, s)}.parquet")
    b["group"] = b.doc_id.str.split("@").str[0]
    b["source"] = s
    frames.append(b)
B = pd.concat(frames).merge(te.reset_index()[["index", "group", "source", "label"]], on=["group", "source"]).set_index("index")
feats = [c for c in B.columns if c.startswith("f_")]
B["proj"] = proj.loc[B.index]
allc = B[feats].corrwith(B.proj).sort_values()
hum = B[B.source == "human"]
humc = hum[feats].corrwith(hum.proj).sort_values()
out["biber"] = dict(all_top_neg=allc.head(8).round(3).to_dict(), all_top_pos=allc.tail(8).round(3).to_dict(),
                    within_human_top_neg=humc.head(8).round(3).to_dict(), within_human_top_pos=humc.tail(8).round(3).to_dict())
print(json.dumps(out["biber"], indent=1))
json.dump(out, open("results/geometry.json", "w"), indent=1)
