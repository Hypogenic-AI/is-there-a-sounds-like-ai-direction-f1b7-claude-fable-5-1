"""Read every corpus text (raw, BOS + text) with a model and save mean-pooled residuals.

usage: python src/extract.py instruct|base
Saves results/acts/{model}_corpus.npy [N, 33, 4096] float16, plus NLL / token counts and a
48-token-window pooled copy for the HAP-E human texts (length direction).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from common import ACTS, DATA, load_model, pooled_acts

key = sys.argv[1]
os.makedirs(ACTS, exist_ok=True)
corpus = pd.read_parquet(f"{DATA}/corpus.parquet")
tok, model = load_model(key)
t0 = time.time()
texts = corpus.text.tolist()
acts, nll, ntok, extra = pooled_acts(tok, model, [None] * len(texts), texts, bs=48, extra_windows=(48,))
np.save(f"{ACTS}/{key}_corpus.npy", acts)
np.save(f"{ACTS}/{key}_corpus_w48.npy", extra[48])
pd.DataFrame(dict(uid=corpus.uid, nll=nll, ntok=ntok)).to_parquet(f"{ACTS}/{key}_corpus_meta.parquet")
print(key, acts.shape, "nll mean", nll.mean(), "ntok mean", ntok.mean(), f"{time.time() - t0:.0f}s")
