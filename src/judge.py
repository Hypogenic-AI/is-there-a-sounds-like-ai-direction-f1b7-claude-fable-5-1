"""LLM judge (OpenRouter) for AI-likeness, coherence and on-task score. Blind to condition.

usage: python src/judge.py results/gens/<name>.jsonl [--conds a,b,c] [--max-pid-rank N]
Writes results/judge/<name>.parquet. Responses are cached in results/judge/cache.jsonl.
"""
import argparse
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

JUDGE_MODEL = "openai/gpt-5.6-terra"
client = OpenAI(api_key=os.environ["OPENROUTER_KEY"], base_url="https://openrouter.ai/api/v1", timeout=25, max_retries=0)
CACHE = "results/judge/cache.jsonl"
os.makedirs("results/judge", exist_ok=True)
lock = threading.Lock()
cache = {}
if os.path.exists(CACHE):
    for l in open(CACHE):
        r = json.loads(l)
        cache[r["key"]] = r

SYSTEM = """You are evaluating a piece of text that was written in response to a writing task. The text was cut off at a fixed length, so ignore the fact that it may end mid-sentence. Rate three things, each as an integer 0-100:

- ai_likelihood: how likely is it that this text was written by an AI language model rather than by a human? 0 = certainly human-written, 100 = certainly AI-generated. Judge by how the text reads (style, phrasing, structure), as an experienced reader would.
- coherence: is the text fluent, grammatical and does it make sense? 0 = gibberish or broken, 50 = understandable but with clear errors, repetition or confusion, 100 = fully coherent. Casual, informal or blunt style is NOT incoherence.
- on_task: does the text do what the task asked for (answer the question / continue the passage / write the requested piece) with relevant content? 0 = unrelated, 100 = fully on task.

Reply with only a JSON object: {"ai_likelihood": int, "coherence": int, "on_task": int}"""


FALLBACK_MODEL = "anthropic/claude-sonnet-5"   # used only when the primary judge's content filter fires


@retry(wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(2))
def _call(model, task, text):
    kw = dict(extra_body={"reasoning": {"effort": "low"}}) if model == JUDGE_MODEL else {}
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=400, **kw,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"<task>\n{task}\n</task>\n\n<text>\n{text}\n</text>"}])
    if r.choices[0].finish_reason == "content_filter":
        return None, r.usage
    c = r.choices[0].message.content
    j = json.loads(re.search(r"\{.*\}", c, re.S).group(0))
    return {k: int(j[k]) for k in ("ai_likelihood", "coherence", "on_task")}, r.usage


def call(task, text):
    res, usage = _call(JUDGE_MODEL, task, text)
    if res is None:
        res, usage = _call(FALLBACK_MODEL, task, text)
        if res is None:
            raise ValueError("content filter on both judges")
        res["fallback"] = 1
    return res, usage


def judge_one(task, text):
    task = " ".join(task.split()[:150])
    text = " ".join(text.split()[:200])
    key = hashlib.md5((JUDGE_MODEL + SYSTEM + task + "\x00" + text).encode()).hexdigest()
    if key in cache:
        return cache[key]
    try:
        res, usage = call(task, text if text else "(empty)")
    except Exception as e:  # leave as missing; reported in the analysis
        print("judge error", repr(e)[:200])
        return dict(ai_likelihood=None, coherence=None, on_task=None)
    rec = dict(key=key, **res, pt=usage.prompt_tokens, ct=usage.completion_tokens)
    with lock:
        cache[key] = rec
        with open(CACHE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def task_text(p):
    if p["task"] == "rawcont":
        return "Continue the following text.\n\n" + " ".join(p["prefix"].split()[-120:])
    if p["task"] == "continue":
        return "Continue the following text in the same voice and style.\n\n" + " ".join(p["prefix"].split()[-120:])
    return p["user"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--conds", default=None)
    ap.add_argument("--npid", type=int, default=None, help="judge only the first N prompts (file order)")
    ap.add_argument("--every", type=int, default=1, help="judge every k-th prompt (stratified subsample)")
    a = ap.parse_args()
    prompts = {p["id"]: p for s in json.load(open("results/data/prompts.json")).values() for p in s}
    df = pd.read_json(a.path, lines=True)
    if a.conds:
        df = df[df.cond.isin(a.conds.split(","))]
    if a.npid:
        keep = list(dict.fromkeys(df.pid))[:a.npid]
        df = df[df.pid.isin(keep)]
    if a.every > 1:
        keep = list(dict.fromkeys(df.pid))[::a.every]
        df = df[df.pid.isin(keep)]
    with ThreadPoolExecutor(48) as ex:
        res = list(ex.map(lambda r: judge_one(task_text(prompts[r[0]]), r[1]), zip(df.pid, df.text)))
    for k in ("ai_likelihood", "coherence", "on_task"):
        df[k] = [r[k] for r in res]
    name = os.path.basename(a.path).replace(".jsonl", "")
    df.drop(columns=["text"]).to_parquet(f"results/judge/{name}.parquet")
    pt = sum(r.get("pt", 0) or 0 for r in res); ct = sum(r.get("ct", 0) or 0 for r in res)
    print(f"judged {len(df)} rows, missing {df.coherence.isna().sum()}, tokens in/out {pt}/{ct}")
