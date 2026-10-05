"""E1 readout + construction of the AI direction and all confound directions.

usage: python src/directions.py instruct|base
Writes results/directions_{model}.npz (each entry [n_layers+1, d], un-normalised
mean differences unless stated) and results/readout_{model}.json.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import ACTS, DATA

key = sys.argv[1]
corpus = pd.read_parquet(f"{DATA}/corpus.parquet")
meta = pd.read_parquet(f"{ACTS}/{key}_corpus_meta.parquet")
corpus["nll"] = meta.nll.values
A = np.load(f"{ACTS}/{key}_corpus.npy", mmap_mode="r")
A48 = np.load(f"{ACTS}/{key}_corpus_w48.npy", mmap_mode="r")
fit = np.load(f"{ACTS}/{key}_fit.npz")
NL = A.shape[1]
INSTRUCT_GENS = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "llama-3-8B-Instruct", "llama-3-70B-Instruct"]
BASE_GENS = ["llama-3-8B", "llama-3-70B"]
RAID_CHAT = ["chatgpt", "gpt4", "llama-chat", "mistral-chat", "mpt-chat", "cohere-chat"]
RAID_NONCHAT = ["gpt2", "gpt3", "mistral", "mpt", "cohere"]


def idx(q):
    return corpus.query(q).index.values


def layer(l, ix, src=A):
    return np.asarray(src[np.sort(ix), l], dtype=np.float32)


def auc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


hp = corpus.dataset == "hape"
tr_h = idx("dataset=='hape' and split=='train' and source=='human'")
tr_ai = idx("dataset=='hape' and split=='train' and source in @INSTRUCT_GENS")
te_h = idx("dataset=='hape' and split=='test' and source=='human'")
te_ai = idx("dataset=='hape' and split=='test' and source in @INSTRUCT_GENS")
genres = sorted(corpus[hp].domain.unique())

D = {k: np.zeros((NL, A.shape[2]), np.float32) for k in
     ["ai_read", "ai_lr", "ai_hc3", "ai_raid", "ai_base_gens", "formality", "fluency", "length", "chat_register",
      "write_raw"]}
D["genre_basis"] = np.zeros((NL, len(genres) - 1, A.shape[2]), np.float32)
if key == "instruct":
    D["write_chat"] = np.zeros_like(D["ai_read"])
    D["assistant_axis"] = np.zeros_like(D["ai_read"])
R = {"layers": {}, "norms": {}}

hc3_groups = sorted(corpus[corpus.dataset == "hc3"].group.unique())
hc3_fit_groups = set(hc3_groups[::2])
raid_groups = sorted(corpus[corpus.dataset == "raid"].group.unique())
raid_fit_groups = set(raid_groups[::2])

for l in range(1, NL):
    r = {}
    Xh, Xa = layer(l, tr_h), layer(l, tr_ai)
    v = Xa.mean(0) - Xh.mean(0)
    D["ai_read"][l] = v
    u = v / np.linalg.norm(v)
    r["resid_norm"] = float(np.linalg.norm(np.r_[Xh, Xa], axis=1).mean())
    r["dm_norm"] = float(np.linalg.norm(v))
    # logistic-regression probe direction (in raw activation space)
    sc = StandardScaler().fit(np.r_[Xh, Xa])
    lr = LogisticRegression(C=0.01, max_iter=300, class_weight="balanced").fit(
        sc.transform(np.r_[Xh, Xa]), np.r_[np.zeros(len(Xh)), np.ones(len(Xa))])
    w = lr.coef_[0] / sc.scale_
    D["ai_lr"][l] = w / np.linalg.norm(w)
    ulr = D["ai_lr"][l]

    def proj_auc(q_pos, q_neg, d=u):
        return auc(layer(l, idx(q_pos)) @ d, layer(l, idx(q_neg)) @ d)

    Th, Ta = layer(l, te_h), layer(l, te_ai)
    r["hape_test_dm"] = auc(Ta @ u, Th @ u)
    r["hape_test_lr"] = auc(Ta @ ulr, Th @ ulr)
    r["cos_dm_lr"] = float(u @ ulr)
    for g in INSTRUCT_GENS + BASE_GENS:
        r[f"hape_gen_{g}"] = auc(layer(l, idx(f"dataset=='hape' and split=='test' and source=='{g}'")) @ u, Th @ u)
    r["hc3_dm"] = proj_auc("dataset=='hc3' and label==1", "dataset=='hc3' and label==0")
    r["hc3_lr"] = proj_auc("dataset=='hc3' and label==1", "dataset=='hc3' and label==0", ulr)
    r["raid_chat_dm"] = proj_auc("dataset=='raid' and source in @RAID_CHAT", "dataset=='raid' and label==0")
    r["raid_nonchat_dm"] = proj_auc("dataset=='raid' and source in @RAID_NONCHAT", "dataset=='raid' and label==0")
    r["raid_chat_lr"] = proj_auc("dataset=='raid' and source in @RAID_CHAT", "dataset=='raid' and label==0", ulr)
    r["raid_nonchat_lr"] = proj_auc("dataset=='raid' and source in @RAID_NONCHAT", "dataset=='raid' and label==0", ulr)
    r["mage_dm"] = proj_auc("dataset=='mage' and label==1", "dataset=='mage' and label==0")
    r["mage_lr"] = proj_auc("dataset=='mage' and label==1", "dataset=='mage' and label==0", ulr)
    # d' of each HAP-E generator along the direction (human sd units)
    sd = (Th @ u).std()
    r["dprime"] = {g: float(((layer(l, idx(f"dataset=='hape' and split=='test' and source=='{g}'")) @ u).mean()
                             - (Th @ u).mean()) / sd) for g in INSTRUCT_GENS + BASE_GENS}

    # ---- directions from other datasets / contrasts
    D["ai_hc3"][l] = (layer(l, idx("dataset=='hc3' and label==1 and group in @hc3_fit_groups")).mean(0)
                      - layer(l, idx("dataset=='hc3' and label==0 and group in @hc3_fit_groups")).mean(0))
    D["ai_raid"][l] = (layer(l, idx("dataset=='raid' and source in @RAID_CHAT and group in @raid_fit_groups")).mean(0)
                       - layer(l, idx("dataset=='raid' and label==0 and group in @raid_fit_groups")).mean(0))
    D["ai_base_gens"][l] = layer(l, idx("dataset=='hape' and split=='train' and source in @BASE_GENS")).mean(0) - Xh.mean(0)
    D["chat_register"][l] = (layer(l, idx("dataset=='hape' and split=='train' and source in ['llama-3-8B-Instruct','llama-3-70B-Instruct']")).mean(0)
                             - layer(l, idx("dataset=='hape' and split=='train' and source in @BASE_GENS")).mean(0))
    D["formality"][l] = (layer(l, idx("dataset=='pavlick' and split=='train' and label==1")).mean(0)
                         - layer(l, idx("dataset=='pavlick' and split=='train' and label==0")).mean(0))
    nll_h = corpus.loc[tr_h, "nll"].values
    lo, hi = np.quantile(nll_h, [1 / 3, 2 / 3])
    D["fluency"][l] = Xh[nll_h <= lo].mean(0) - Xh[nll_h >= hi].mean(0)      # low NLL minus high NLL
    D["length"][l] = Xh.mean(0) - layer(l, tr_h, A48).mean(0)                  # 256-token minus 48-token window
    gm = np.stack([layer(l, idx(f"dataset=='hape' and split=='train' and source=='human' and domain=='{g}'")).mean(0)
                   for g in genres])
    gm = gm - gm.mean(0)
    _, _, Vt = np.linalg.svd(gm, full_matrices=False)
    D["genre_basis"][l] = Vt[:len(genres) - 1]
    D["write_raw"][l] = fit["rawcont_self"][:, l].astype(np.float32).mean(0) - fit["rawcont_human"][:, l].astype(np.float32).mean(0)
    if key == "instruct":
        D["write_chat"][l] = fit["chat_self"][:, l].astype(np.float32).mean(0) - fit["chat_human"][:, l].astype(np.float32).mean(0)
        D["assistant_axis"][l] = fit["axis_default"][l] - fit["axis_roles"][:, l].mean(0)

    # cross-dataset transfer of the other directions onto held-out halves and onto HAP-E
    uh, ur = D["ai_hc3"][l] / np.linalg.norm(D["ai_hc3"][l]), D["ai_raid"][l] / np.linalg.norm(D["ai_raid"][l])
    r["hc3dir_on_hc3_heldout"] = proj_auc("dataset=='hc3' and label==1 and group not in @hc3_fit_groups",
                                          "dataset=='hc3' and label==0 and group not in @hc3_fit_groups", uh)
    r["hc3dir_on_hape"] = auc(Ta @ uh, Th @ uh)
    r["raiddir_on_hape"] = auc(Ta @ ur, Th @ ur)
    r["hapedir_on_hc3_heldout"] = proj_auc("dataset=='hc3' and label==1 and group not in @hc3_fit_groups",
                                           "dataset=='hc3' and label==0 and group not in @hc3_fit_groups")
    # readout of confounds along the AI direction and vice versa
    r["formality_auc_along_ai"] = proj_auc("dataset=='pavlick' and split=='test' and label==1",
                                           "dataset=='pavlick' and split=='test' and label==0")
    uf = D["formality"][l] / np.linalg.norm(D["formality"][l])
    r["formality_auc_along_formality"] = proj_auc("dataset=='pavlick' and split=='test' and label==1",
                                                  "dataset=='pavlick' and split=='test' and label==0", uf)
    r["ai_auc_along_formality"] = auc(Ta @ uf, Th @ uf)

    # cosines
    cos = {}
    for k, vv in D.items():
        if k in ("ai_read", "genre_basis"):
            continue
        cos[k] = float(u @ (vv[l] / (np.linalg.norm(vv[l]) + 1e-9)))
    cos["genre_subspace_frac"] = float(((D["genre_basis"][l] @ u) ** 2).sum())
    r["cos_with_ai_read"] = cos

    # leave-one-genre-out
    logo = {}
    for g in genres:
        oth_h = idx(f"dataset=='hape' and split=='train' and source=='human' and domain!='{g}'")
        oth_a = idx(f"dataset=='hape' and split=='train' and source in @INSTRUCT_GENS and domain!='{g}'")
        vg = layer(l, oth_a).mean(0) - layer(l, oth_h).mean(0)
        logo[g] = auc(layer(l, idx(f"dataset=='hape' and split=='test' and source in @INSTRUCT_GENS and domain=='{g}'")) @ vg,
                      layer(l, idx(f"dataset=='hape' and split=='test' and source=='human' and domain=='{g}'")) @ vg)
    r["leave_one_genre_out"] = logo

    # projection-out: remove confound span from activations, refit diff-of-means, held-out AUROC
    conf = ["formality", "fluency", "length", "chat_register"] + (["assistant_axis"] if key == "instruct" else [])
    po = {}

    def po_auc(B):
        Q, _ = np.linalg.qr(np.asarray(B, np.float64).T)
        Q = Q.astype(np.float32)
        rm = lambda X: X - (X @ Q) @ Q.T
        vv = rm(Xa).mean(0) - rm(Xh).mean(0)
        return auc(rm(Ta) @ vv, rm(Th) @ vv), float(1 - np.linalg.norm(u - (u @ Q) @ Q.T) ** 2)

    for c in conf:
        po[c] = po_auc([D[c][l]])
    po["genre"] = po_auc(list(D["genre_basis"][l]))
    po["all"] = po_auc([D[c][l] for c in conf] + list(D["genre_basis"][l]))
    r["projection_out"] = {k: {"auroc": a, "frac_ai_dir_removed": f} for k, (a, f) in po.items()}
    R["layers"][l] = r
    print(l, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

np.savez(f"results/directions_{key}.npz", **D)
json.dump(R, open(f"results/readout_{key}.json", "w"), indent=1)
