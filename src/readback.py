"""Closed-loop check and logit lens.

usage: python src/readback.py <model key> <gens name> <layer>
(1) Re-read every generated text (raw, BOS + text) with the *unsteered* model and project the
    mean-pooled activations onto the unit AI readout direction at <layer>. Not an independent
    detector; it asks whether steering writes into the text what the probe reads from text.
    -> results/scores/<gens>_readback.parquet (proj in units of human-text sd above human mean)
(2) Logit lens of the direction: tokens most promoted / suppressed by +direction.
    -> results/logit_lens_<model>.json
"""
import json
import sys

import numpy as np
import pandas as pd
import torch

from common import ACTS, DATA, load_model, pooled_acts, unit

key, name, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok, model = load_model(key)
D = np.load(f"results/directions_{key}.npz")
u = unit(D["ai_read"][L]).astype(np.float32)

# human-text reference scale from HAP-E test human texts
corpus = pd.read_parquet(f"{DATA}/corpus.parquet")
ix = np.sort(corpus.query("dataset=='hape' and split=='test' and source=='human'").index.values)
ph = np.asarray(np.load(f"{ACTS}/{key}_corpus.npy", mmap_mode="r")[ix, L], np.float32) @ u
mu, sd = ph.mean(), ph.std()

for nm in name.split(","):
    df = pd.read_json(f"results/gens/{nm}.jsonl", lines=True)
    texts = [t if len(t.strip()) else "." for t in df.text]
    acts, nll, ntok = pooled_acts(tok, model, [None] * len(texts), texts, bs=64)
    df["readback"] = (acts[:, L].astype(np.float32) @ u - mu) / sd
    df["self_nll"] = nll
    df.drop(columns=["text"]).to_parquet(f"results/scores/{nm}_readback.parquet")
    print(nm, df.groupby("cond", sort=False).readback.mean().round(2).to_dict())

# logit lens: direction through final norm + unembedding
with torch.no_grad():
    W = model.lm_head.weight.float()                                  # [V, d]
    lens = {}
    for dname in ("ai_read", "chat_register", "formality", "write_chat", "assistant_axis"):
        if dname not in D:
            continue
        v = torch.tensor(unit(D[dname][L]), dtype=torch.float32, device="cuda")
        s = W @ v
        top, bot = s.topk(40).indices.tolist(), (-s).topk(40).indices.tolist()
        lens[dname] = dict(promoted=[tok.decode([i]) for i in top], suppressed=[tok.decode([i]) for i in bot])
json.dump(lens, open(f"results/logit_lens_{key}.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(lens.get("ai_read"), ensure_ascii=False)[:1500])
