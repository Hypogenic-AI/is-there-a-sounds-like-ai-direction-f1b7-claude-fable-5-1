"""Shared utilities: model loading, pooled-activation extraction, steering hooks, generation."""
import contextlib
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "base": "meta-llama/Llama-3.1-8B",
}
MAX_TEXT_TOKENS = 256      # tokens of each text that are read / pooled
MAX_NEW_TOKENS = 200       # tokens generated per prompt
GEN_KW = dict(do_sample=True, temperature=0.7, top_p=0.95)
DATA = "results/data"
ACTS = "results/acts"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(key):
    name = MODELS.get(key, key)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def load_prompts():
    return json.load(open(f"{DATA}/prompts.json"))


def chat_prefix(tok, user, system=None):
    """Chat-template string up to (and including) the assistant header."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def pooled_acts(tok, model, prefixes, texts, bs=32, max_text_tokens=MAX_TEXT_TOKENS, extra_windows=()):
    """Mean-pool the residual stream over the tokens of `texts`, at every layer.

    prefixes[i] is a context string (already containing BOS / chat template, or None for
    a bare BOS) that is *not* pooled over; texts[i] is the text whose tokens are pooled.
    Returns acts [N, n_layers+1, d] (float16), nll [N] (mean token NLL of the text given
    the prefix), ntok [N], and for each w in extra_windows acts pooled over the first w
    text tokens only.
    """
    N = len(texts)
    out, nlls, ntoks = None, np.zeros(N, np.float32), np.zeros(N, np.int32)
    extra = {w: None for w in extra_windows}
    order = np.argsort([len(t) + len(p or "") for p, t in zip(prefixes, texts)])
    for b in range(0, N, bs):
        idx = order[b:b + bs]
        seqs, starts = [], []
        for i in idx:
            p = prefixes[i]
            pid = [tok.bos_token_id] if p is None else tok(p, add_special_tokens=False).input_ids
            tid = tok(texts[i], add_special_tokens=False).input_ids[:max_text_tokens]
            seqs.append(pid + tid)
            starts.append(len(pid))
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        tmask = torch.zeros((len(seqs), L))
        for j, s in enumerate(seqs):  # right-pad here (no generation), so positions are aligned from 0
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
            tmask[j, starts[j]:len(s)] = 1
        ids, att, tmask = ids.cuda(), att.cuda(), tmask.cuda()
        o = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        nl = len(o.hidden_states)
        denom = tmask.sum(1)[:, None]
        wms = {}
        for w in extra:
            wm = tmask.clone()
            for j in range(len(seqs)):
                wm[j, starts[j] + w:] = 0
            wms[w] = wm
        if out is None:
            d = o.hidden_states[0].shape[-1]
            out = np.zeros((N, nl, d), np.float16)
            for w in extra:
                extra[w] = np.zeros_like(out)
        for l in range(nl):  # pool layer by layer to keep memory low
            h = o.hidden_states[l].float()
            out[idx, l] = ((h * tmask[..., None]).sum(1) / denom).cpu().numpy().astype(np.float16)
            for w, wm in wms.items():
                extra[w][idx, l] = ((h * wm[..., None]).sum(1) / wm.sum(1)[:, None]).cpu().numpy().astype(np.float16)
        # mean NLL of text tokens
        for j, i in enumerate(idx):  # per sample, text tokens only (keeps memory low)
            a, e = starts[j], len(seqs[j])
            nlls[i] = torch.nn.functional.cross_entropy(o.logits[j, a - 1:e - 1].float(), ids[j, a:e]).item()
        ntoks[idx] = tmask.sum(1).cpu().numpy()
        del o
    return (out, nlls, ntoks, extra) if extra_windows else (out, nlls, ntoks)


# ----------------------------------------------------------------------------- steering
@contextlib.contextmanager
def steering(model, add=None, ablate=None):
    """Context manager installing residual-stream interventions.

    add:    list of (hidden_state_index l, vector [d]) -> vector is added to the output of
            block l-1 (i.e. hidden_states[l]) at every position.
    ablate: unit vector [d]; its component is projected out of the output of every block.
    """
    handles = []
    layers = model.model.layers
    dt = next(model.parameters()).dtype

    def mk_add(v):
        v = v.to("cuda", dt)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + v
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def mk_abl(u):
        u = (u / u.norm()).to("cuda", dt)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - (h @ u)[..., None] * u
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    try:
        if ablate is not None:
            for lyr in layers:
                handles.append(lyr.register_forward_hook(mk_abl(torch.as_tensor(ablate))))
        for l, v in (add or []):
            handles.append(layers[l - 1].register_forward_hook(mk_add(torch.as_tensor(v))))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate(tok, model, contexts, bs=104, seed=0, max_new_tokens=MAX_NEW_TOKENS):
    """Sample one completion per context string (context already includes BOS / template)."""
    outs = [None] * len(contexts)
    order = np.argsort([len(c) for c in contexts])
    for b in range(0, len(contexts), bs):
        idx = order[b:b + bs]
        enc = tok([contexts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        set_seed(seed * 1000 + b)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **GEN_KW)
        dec = tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i, d in zip(idx, dec):
            outs[i] = d.strip()
    return outs


def make_contexts(tok, prompts, system=None, suffix=""):
    """Context strings for generation. Chat tasks use the chat template; rawcont uses BOS + prefix."""
    ctx = []
    for p in prompts:
        if p["task"] == "rawcont":
            ctx.append(tok.bos_token + p["prefix"])
        else:
            ctx.append(chat_prefix(tok, p["user"] + suffix, system))
    return ctx


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)
