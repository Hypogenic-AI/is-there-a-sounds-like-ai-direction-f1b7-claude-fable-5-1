# Downloaded Datasets

Data files are NOT committed to git (see `.gitignore`). Everything here is
reproduced by one script, run from the workspace root:

```bash
source .venv/bin/activate
python datasets/download_datasets.py            # all
python datasets/download_datasets.py hc3 raid   # subset: hc3 hape mage raid formality assistant_axis
```

Small truncated samples are in each `*/samples/samples.json`.

| Dir | Source | Rows | On disk |
|---|---|---|---|
| `hc3/` | `Hello-SimpleAI/HC3` | 24,322 questions | 147 MB |
| `human_ai_parallel_corpus/` | `browndw/human-ai-parallel-corpus` | 8,290 docs × 8 sources | 114 MB |
| `human_ai_parallel_corpus_biber/` | `browndw/human-ai-parallel-corpus-biber` | 8,290 × 67 features × 8 sources | 13 MB |
| `mage/` | `yaful/MAGE` | 319k / 57k / 57k (+2 OOD sets) | 554 MB |
| `raid/` | `liamdugan/raid` (train.csv, `attack == none` only) | see below | see below |
| `pavlick_formality/` | `osyvokon/pavlick-formality-scores` | 11,274 sentences | 2 MB |
| `assistant_axis_vectors/` | `lu-christina/assistant-axis-vectors` (axis files only) | 3 models | 5 MB |

## 1. HC3 (Human ChatGPT Comparison Corpus)
- **Task**: paired human vs ChatGPT answers to the same question. License CC-BY-SA.
- **Format**: JSONL; `question`, `human_answers` (list, mean 2.4), `chatgpt_answers` (list, mean 1.1), `source`. `all.jsonl` = union of `reddit_eli5` (17,112), `finance` (3,933), `medicine` (1,248), `open_qa` (1,187), `wiki_csai` (842).
- **Load**: `pd.read_json("datasets/hc3/all.jsonl", lines=True)`
- **Notes**: strongly confounded — human side is casual Reddit/forum text, ChatGPT (2022, gpt-3.5) is formal assistant prose; first answers average 147 (human) vs 175 (ChatGPT) words. Good for an easy direction and for *question prompts* to generate from; bad as the only source for a de-confounded direction. Some ChatGPT answers are empty lists.

## 2. HAP-E (Human-AI Parallel English corpus) — recommended primary source for the direction
- **Task**: each LLM was given human chunk 1 (~500 words) and asked to continue ~500 words; compare with human chunk 2 (the true continuation). Topic/domain/length are therefore matched by construction. License MIT.
- **Format**: parquet, columns `doc_id` (`<genre>_<n>@<source>`), `text`. Sources: `human-chunk-1` (the prompt; exclude from classification), `human-chunk-2`, `gpt-4o-2024-08-06`, `gpt-4o-mini-2024-07-18`, `llama-3-8B`, `llama-3-8B-Instruct`, `llama-3-70B`, `llama-3-70B-Instruct`. 8,290 docs per source; genres: spok 1721, blog 1526, fic 1395, news 1322, acad 1227, tvm 1099.
- **Load**: `pd.read_parquet("datasets/human_ai_parallel_corpus/text_data/hape-text_human-chunk-2.parquet")`; join across sources on `doc_id.split("@")[0]`.
- **Notes**: mean words: human 479, gpt-4o 528, gpt-4o-mini 578, llama-8B-Instruct 426, llama-8B base 480. Contains **base and instruct** variants of the same Llama models — directly supports the base-vs-instruct (assistant persona) contrast. Base-model outputs can be noisy/repetitive.
- **Biber companion** (`human_ai_parallel_corpus_biber/biber_data/*.parquet`): 67 Biber lexico-grammatical feature rates per doc (`f_01_past_tense` …), same `doc_id`s. Use as interpretable style covariates (nominalisations, contractions, pronouns, etc.) to regress against the direction projection.

## 3. MAGE (a.k.a. DeepfakeTextDetect)
- **Task**: binary detection, 10 domains × 27 generators. License Apache-2.0.
- **Format**: CSV `text`, `label`, `src`. **`label` 1 = human, 0 = machine** (note the inversion). `src` encodes domain + generator, e.g. `cmv_human`, `xsum_machine_continuation_flan_t5_small`.
- **Splits**: train 319,071 · valid 56,792 · test 56,819 · `test_ood_set_gpt.csv` 1,562 (unseen domain + GPT-4) · `test_ood_set_gpt_para.csv` 2,362 (paraphrased).
- **Load**: `pd.read_csv("datasets/mage/test.csv")`
- **Notes**: many generators are old/small (flan-t5, GLM, OPT, GPT-3 davinci-002/003, gpt-3.5-turbo). Use for transfer tests of the direction, and filter `src` for gpt-3.5 if a modern-style subset is wanted.

## 4. RAID (no-attack subset of train)
- **Task**: detection benchmark; 11 generators (chat and non-chat pairs: llama-chat, mistral / mistral-chat, mpt / mpt-chat, cohere / cohere-chat, gpt2, gpt3, chatgpt, gpt4) + human, 8 domains (abstracts, books, news, poetry, recipes, reddit, reviews, wiki), 2 decodings × repetition penalty. License MIT.
- **Format**: CSV; `id`, `adv_source_id`, `source_id` (id of the human doc the prompt came from → pair human/AI on this), `model`, `decoding`, `repetition_penalty`, `attack`, `domain`, `title`, `prompt`, `generation` (the text; for `model == human` it is the human document).
- **Download**: the full `train.csv` is 11.8 GB; `download_datasets.py raid` streams it and keeps only `attack == "none"` rows into `raid/train_none.csv`. (The 1.2 GB `test.csv` has hidden labels; `extra.csv` adds code/German/Czech — not downloaded.)
- **Load**: `pd.read_csv("datasets/raid/train_none.csv")`
- **Notes**: `title` + `prompt` give ready-made generation prompts with a matched human document; chat vs non-chat generator pairs give a second base-vs-chat contrast. Row counts: see "Validation" below.

## 5. Pavlick & Tetreault formality scores (confound control)
- 11,274 sentences (`domain` ∈ answers/blog/email/news, `avg_score` ∈ [−3, 3], `sentence`). Use to build an independent *formality* direction (diff-of-means of top vs bottom tercile, or regression) for cosine/projection-out comparisons. A ready classifier `s-nlp/roberta-base-formality-ranker` is on HF for scoring generated text.

## 6. Assistant Axis vectors (confound control)
- `<model>/assistant_axis.pt`, `default_vector.pt`: tensors `[n_layers, d_model]` — gemma-2-27b `[46, 4608]`, qwen-3-32b `[64, 5120]`, llama-3.3-70b `[80, 8192]`; `capping_config.pt` (dict) for Qwen and Llama. Role/trait vectors (1.2 GB) were **not** downloaded — see comment in `download_datasets.py`.
- **Load**: `torch.load("datasets/assistant_axis_vectors/gemma-2-27b/assistant_axis.pt")`
- **Notes**: only usable directly if the experiment model is one of these three (Gemma-2-27B-it in bf16 ≈ 54 GB does not fit the 48 GB A6000 without quantisation; Qwen3-32B likewise). For a smaller model, recompute the axis with `code/assistant-axis/pipeline` or a cheap proxy (see `literature_review.md`).

## Validation
All files were loaded with pandas/torch after download; schemas and counts above are from that check. RAID counts are appended below once the stream completes.
