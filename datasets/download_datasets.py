"""Download all datasets used in this project into datasets/.

Usage (from workspace root):  python datasets/download_datasets.py
RAID is 16.7 GB in full; we stream train.csv and keep only the attack == 'none'
rows (see raid()).
"""
import csv, io, os, sys
import requests
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))

def hc3():
    snapshot_download("Hello-SimpleAI/HC3", repo_type="dataset", local_dir=f"{ROOT}/hc3",
                      allow_patterns=["*.jsonl", "README.md"])

def hape():
    snapshot_download("browndw/human-ai-parallel-corpus", repo_type="dataset",
                      local_dir=f"{ROOT}/human_ai_parallel_corpus")
    snapshot_download("browndw/human-ai-parallel-corpus-biber", repo_type="dataset",
                      local_dir=f"{ROOT}/human_ai_parallel_corpus_biber")

def mage():
    snapshot_download("yaful/MAGE", repo_type="dataset", local_dir=f"{ROOT}/mage",
                      allow_patterns=["*.csv", "README.md"])

def raid():
    """Stream the full RAID train.csv (11.8 GB) but keep only attack == 'none' rows
    (all 8 domains, 11 generators + human; ~1/11 of the file)."""
    os.makedirs(f"{ROOT}/raid", exist_ok=True)
    out = f"{ROOT}/raid/train_none.csv"
    url = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
    hdr = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    csv.field_size_limit(sys.maxsize)
    with requests.get(url, stream=True, headers=hdr) as r, open(out, "w", newline="") as f:
        r.raise_for_status()
        r.raw.decode_content = True
        reader = csv.reader(io.TextIOWrapper(r.raw, encoding="utf-8", newline=""))
        w = csv.writer(f)
        header = next(reader)
        w.writerow(header)
        k = header.index("attack")
        total = int(r.headers.get("Content-Length", 0))
        try:
            for row in reader:
                if row[k] == "none":
                    w.writerow(row)
        except ValueError:
            # urllib3 closes the raw stream at EOF, which TextIOWrapper reports as
            # "I/O operation on closed file"; only accept it if every byte arrived.
            pass
        got = r.raw.tell()
        assert total == 0 or got == total, f"RAID stream truncated: {got} of {total} bytes"
        print(f"raid: read {got} of {total} bytes")

def formality():
    snapshot_download("osyvokon/pavlick-formality-scores", repo_type="dataset",
                      local_dir=f"{ROOT}/pavlick_formality", allow_patterns=["*.csv", "README.md"])

def assistant_axis():
    # axis / default vectors / capping configs only (role+trait vectors are 1.2 GB; add
    # "*/role_vectors/*" to allow_patterns if persona-space PCA is needed)
    snapshot_download("lu-christina/assistant-axis-vectors", repo_type="dataset",
                      local_dir=f"{ROOT}/assistant_axis_vectors",
                      allow_patterns=["README.md", "*/assistant_axis.pt", "*/default_vector.pt", "*/capping_config.pt"])

if __name__ == "__main__":
    for fn in (hc3, hape, mage, raid, formality, assistant_axis):
        if len(sys.argv) > 1 and fn.__name__ not in sys.argv[1:]:
            continue
        print("downloading", fn.__name__, flush=True)
        fn()
