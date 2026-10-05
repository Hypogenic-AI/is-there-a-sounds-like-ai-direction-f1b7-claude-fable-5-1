"""Score generated / reference texts with detectors that are independent of Llama activations.

usage: python src/score.py results/gens/<name>.jsonl [...]
For each input file writes results/scores/<name>.parquet with one row per text:
  desklib      P(AI) from desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, supervised)
  radar        P(AI) from TrustSafeAI/RADAR-Vicuna-7B (RoBERTa-large, adversarially trained)
  hc3rob       P(ChatGPT) from Hello-SimpleAI/chatgpt-detector-roberta
  bino         Binoculars-style score, sign flipped so that higher = more AI-like
               (observer Qwen2.5-7B, performer Qwen2.5-7B-Instruct)
  ext_nll      mean token NLL under Qwen2.5-7B (external fluency measure)
  formality    P(formal) from s-nlp/roberta-base-formality-ranker
  distinct3    distinct trigram ratio (repetition), nwords
Embeddings (all-mpnet-base-v2) are saved to results/scores/<name>_emb.npy.
All texts are scored on their first 200 whitespace words.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

DEV = "cuda"


class DesklibAIDetectionModel(nn.Module):
    """Architecture from the desklib/ai-text-detector-v1.01 model card (DeBERTa-v3-large encoder,
    masked mean pooling, linear head). Weights are loaded manually from the checkpoint."""

    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, name):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        m = cls(AutoConfig.from_pretrained(name))
        sd = load_file(hf_hub_download(name, "model.safetensors"))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if "position_ids" not in k], missing
        return m

    def forward(self, input_ids, attention_mask=None):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


def batches(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i:i + bs]


@torch.no_grad()
def score_desklib(texts):
    name = "desklib/ai-text-detector-v1.01"
    tok = AutoTokenizer.from_pretrained(name)
    m = DesklibAIDetectionModel.from_pretrained(name).to(DEV).eval()
    out = []
    for b in batches(texts, 64):
        enc = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        out += torch.sigmoid(m(enc.input_ids, enc.attention_mask))[:, 0].float().cpu().tolist()
    del m
    return out


@torch.no_grad()
def score_seqcls(texts, name, ai_index):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(DEV).eval()
    out = []
    for b in batches(texts, 64):
        enc = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        out += torch.softmax(m(**enc).logits.float(), -1)[:, ai_index].cpu().tolist()
    del m
    return out


@torch.no_grad()
def score_binoculars(texts, observer="Qwen/Qwen2.5-7B", performer="Qwen/Qwen2.5-7B-Instruct"):
    """Binoculars (Hans et al. 2024): log-PPL under the performer / cross-entropy between
    observer and performer next-token distributions. Low ratio = machine; we return -ratio."""
    tok = AutoTokenizer.from_pretrained(observer)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    obs = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=torch.bfloat16, device_map=DEV).eval()
    per = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=torch.bfloat16, device_map=DEV).eval()
    bino, nll_obs = [], []
    for b in batches(texts, 8):
        enc = tok(b, padding=True, truncation=True, max_length=384, return_tensors="pt").to(DEV)
        lo = obs(**enc).logits[:, :-1].float()
        lp = per(**enc).logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        m = enc.attention_mask[:, 1:].float()
        ce_p = torch.nn.functional.cross_entropy(lp.transpose(1, 2), tgt, reduction="none")
        ce_o = torch.nn.functional.cross_entropy(lo.transpose(1, 2), tgt, reduction="none")
        x = -(torch.softmax(lo, -1) * torch.log_softmax(lp, -1)).sum(-1)
        ppl = (ce_p * m).sum(1) / m.sum(1)
        xppl = (x * m).sum(1) / m.sum(1)
        bino += (-(ppl / xppl)).cpu().tolist()
        nll_obs += ((ce_o * m).sum(1) / m.sum(1)).cpu().tolist()
    del obs, per
    return bino, nll_obs


def distinct3(t):
    w = t.lower().split()
    tri = list(zip(w, w[1:], w[2:]))
    return len(set(tri)) / max(len(tri), 1)


def main(path):
    name = os.path.basename(path).replace(".jsonl", "")
    out = f"results/scores/{name}.parquet"
    os.makedirs("results/scores", exist_ok=True)
    df = pd.read_json(path, lines=True)
    texts = [" ".join(str(t).split()[:200]) or "." for t in df.text]
    df["nwords"] = [len(str(t).split()) for t in df.text]
    df["distinct3"] = [distinct3(t) for t in texts]
    df["desklib"] = score_desklib(texts); torch.cuda.empty_cache()
    df["radar"] = score_seqcls(texts, "TrustSafeAI/RADAR-Vicuna-7B", 0); torch.cuda.empty_cache()
    df["hc3rob"] = score_seqcls(texts, "Hello-SimpleAI/chatgpt-detector-roberta", 1); torch.cuda.empty_cache()
    df["formality"] = score_seqcls(texts, "s-nlp/roberta-base-formality-ranker", 1); torch.cuda.empty_cache()
    df["bino"], df["ext_nll"] = score_binoculars(texts); torch.cuda.empty_cache()
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEV)
    emb = st.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    np.save(f"results/scores/{name}_emb.npy", emb.astype(np.float32))
    df.drop(columns=["text"]).to_parquet(out)
    print("scored", path, len(df))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
