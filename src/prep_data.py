"""Build the text corpora and generation prompts used by all experiments.

Outputs (results/data/):
  corpus.parquet   one row per text to be *read* by the models
                   columns: uid, dataset, split, group (pair id), domain, source, label (1 = AI), text
  prompts.json     generation prompts (dev / test) with a matched human reference
"""
import json
import os
import re

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
OUT = "results/data"
os.makedirs(OUT, exist_ok=True)
MAXW = 320  # words kept per text (the models read at most 256 tokens anyway)


def clip(t, n=MAXW):
    return " ".join(str(t).split()[:n])


rows = []

# ---------------------------------------------------------------- HAP-E
HAPE = "datasets/human_ai_parallel_corpus/text_data/hape-text_{}.parquet"
GENS = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "llama-3-8B-Instruct",
        "llama-3-70B-Instruct", "llama-3-8B", "llama-3-70B"]
hape = {}
for s in ["human-chunk-1", "human-chunk-2"] + GENS:
    d = pd.read_parquet(HAPE.format(s))
    d["doc"] = d.doc_id.str.split("@").str[0]
    hape[s] = d.set_index("doc").text
docs = sorted(set.intersection(*[set(v.index) for v in hape.values()]))
# keep docs where every source has a reasonably long text
docs = [d for d in docs if all(len(str(hape[s][d]).split()) >= 150 for s in hape)]
by_genre = {}
for d in docs:
    by_genre.setdefault(d.split("_")[0], []).append(d)
N_TRAIN, N_TEST, N_GEN_DEV, N_GEN_TEST = 200, 100, 8, 14  # per genre
hape_gen = {"dev": [], "test": [], "fit": [], "cont_fit": [], "cont_test": []}
for g, ds in sorted(by_genre.items()):
    ds = list(rng.permutation(ds))
    tr, te = ds[:N_TRAIN], ds[N_TRAIN:N_TRAIN + N_TEST]
    rest = ds[N_TRAIN + N_TEST:]
    hape_gen["dev"] += rest[:N_GEN_DEV]
    hape_gen["test"] += rest[N_GEN_DEV:N_GEN_DEV + N_GEN_TEST]
    hape_gen["fit"] += rest[22:52]
    hape_gen["cont_fit"] += rest[52:82]
    hape_gen["cont_test"] += rest[82:107]
    for split, dd in [("train", tr), ("test", te)]:
        for d in dd:
            rows.append(dict(dataset="hape", split=split, group=d, domain=g, source="human",
                             label=0, text=clip(hape["human-chunk-2"][d])))
            for s in GENS:
                rows.append(dict(dataset="hape", split=split, group=d, domain=g, source=s,
                                 label=1, text=clip(hape[s][d])))
print("HAP-E genres", {g: len(v) for g, v in by_genre.items()})

# ---------------------------------------------------------------- HC3
hc3 = pd.read_json("datasets/hc3/all.jsonl", lines=True)
hc3 = hc3[hc3.source != "wiki_csai"]
hc3 = hc3[hc3.human_answers.map(lambda a: len(a) > 0 and len(a[0].split()) >= 60)
          & hc3.chatgpt_answers.map(lambda a: len(a) > 0 and len(a[0].split()) >= 60)]
hc3_gen = {"dev": [], "test": [], "fit": []}
for src, d in hc3.groupby("source"):
    d = d.sample(n=min(len(d), 280), random_state=SEED)
    for i, (_, r) in enumerate(d.iterrows()):
        gid = f"hc3_{src}_{i}"
        if i < 200:
            for lab, t, s in [(0, r.human_answers[0], "human"), (1, r.chatgpt_answers[0], "chatgpt")]:
                rows.append(dict(dataset="hc3", split="test", group=gid, domain=src, source=s,
                                 label=lab, text=clip(t)))
        elif i < 207:
            hc3_gen["dev"].append(dict(id=gid, q=r.question, ref=r.human_answers[0], domain=src))
        elif i >= 230:
            hc3_gen["fit"].append(dict(id=gid, q=r.question, ref=r.human_answers[0], domain=src))
        else:
            hc3_gen["test"].append(dict(id=gid, q=r.question, ref=r.human_answers[0], domain=src))

# ---------------------------------------------------------------- RAID
raid = pd.read_csv("datasets/raid/train_none.csv")
raid = raid[(raid.model == "human") | ((raid.decoding == "sampling") & (raid.repetition_penalty == "no"))]
raid_gen = {"dev": [], "test": [], "fit": []}
CHAT_PROMPT_MODEL = "chatgpt"
for dom, d in raid.groupby("domain"):
    hum = d[d.model == "human"]
    hum = hum[hum.generation.map(lambda t: len(str(t).split()) >= 120)]
    sids = list(rng.permutation(hum.source_id.values))
    read_ids, gen_ids = sids[:60], sids[60:100]
    sub = d[d.source_id.isin(read_ids)]
    for _, r in sub.iterrows():
        if len(str(r.generation).split()) < 40:
            continue
        rows.append(dict(dataset="raid", split="test", group=r.source_id, domain=dom, source=r.model,
                         label=int(r.model != "human"), text=clip(r.generation)))
    if dom in ("news", "reviews", "reddit", "wiki", "books"):  # prose domains for generation
        for j, sid in enumerate(gen_ids):
            p = d[(d.source_id == sid) & (d.model == CHAT_PROMPT_MODEL)].prompt.iloc[0]
            ref = hum[hum.source_id == sid].generation.iloc[0]
            raid_gen["dev" if j < 4 else ("test" if j < 15 else "fit")].append(dict(id=f"raid_{dom}_{j}", q=p, ref=ref, domain=dom))

# ---------------------------------------------------------------- MAGE
mage = pd.read_csv("datasets/mage/test.csv")
mage = mage[mage.text.map(lambda t: len(str(t).split()) >= 100)]
mh = mage[mage.label == 1].sample(800, random_state=SEED)
mm = mage[mage.label == 0].sample(800, random_state=SEED)
for lab, d in [(0, mh), (1, mm)]:
    for i, r in d.reset_index().iterrows():
        dom = re.sub(r"_(human|machine).*$", "", r.src)
        gen = "human" if lab == 0 else re.sub(r"^.*_machine_", "", r.src)
        rows.append(dict(dataset="mage", split="test", group=f"mage_{lab}_{i}", domain=dom, source=gen,
                         label=lab, text=clip(r.text)))

# ---------------------------------------------------------------- Pavlick formality (sentences)
pav = pd.read_csv("datasets/pavlick_formality/all.csv")
pav = pav[pav.sentence.map(lambda s: 8 <= len(str(s).split()) <= 60)]
lo, hi = pav.avg_score.quantile(0.2), pav.avg_score.quantile(0.8)
for lab, d in [(0, pav[pav.avg_score <= lo]), (1, pav[pav.avg_score >= hi])]:
    d = d.sample(min(len(d), 1200), random_state=SEED)
    for i, r in d.reset_index().iterrows():
        rows.append(dict(dataset="pavlick", split="train" if i % 4 else "test", group=f"pav_{lab}_{i}",
                         domain=r.domain, source="formal" if lab else "informal", label=lab,
                         text=str(r.sentence)))

corpus = pd.DataFrame(rows)
corpus["group"] = corpus.group.astype(str)
corpus.insert(0, "uid", np.arange(len(corpus)))
corpus.to_parquet(f"{OUT}/corpus.parquet")
print(corpus.groupby(["dataset", "split", "label"]).size())

# ---------------------------------------------------------------- generation prompts
def tail_words(t, n):
    return " ".join(str(t).split()[-n:])


prompts = {"dev": [], "test": [], "fit": []}
for split in ("dev", "test", "fit"):
    for d in hape_gen[split]:
        prefix = tail_words(hape["human-chunk-1"][d], 220)
        prompts[split].append(dict(
            id=f"hape_{d}", task="continue", domain=d.split("_")[0],
            user=("Continue the following text in the same voice and style. Write only the "
                  f"continuation, about 150 words.\n\n{prefix}"),
            prefix=prefix, ref=clip(hape["human-chunk-2"][d], 200)))
    for r in hc3_gen[split]:
        prompts[split].append(dict(id=r["id"], task="answer", domain=r["domain"], user=r["q"].strip(),
                                   prefix=None, ref=clip(r["ref"], 200)))
    for r in raid_gen[split]:
        prompts[split].append(dict(id=r["id"], task="write", domain=r["domain"], user=r["q"].strip(),
                                   prefix=None, ref=clip(r["ref"], 200)))
# raw-continuation prompts (no chat template) for the base-vs-instruct experiment
for split in ("cont_fit", "cont_test"):
    prompts[split] = [dict(id=f"hape_{d}", task="rawcont", domain=d.split("_")[0], user=None,
                           prefix=tail_words(hape["human-chunk-1"][d], 220),
                           ref=clip(hape["human-chunk-2"][d], 200)) for d in hape_gen[split]]
json.dump(prompts, open(f"{OUT}/prompts.json", "w"), indent=1)
for s in prompts:
    print(s, len(prompts[s]), pd.Series([p["task"] for p in prompts[s]]).value_counts().to_dict())
