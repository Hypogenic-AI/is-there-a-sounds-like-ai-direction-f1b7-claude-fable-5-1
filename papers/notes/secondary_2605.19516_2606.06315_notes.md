# Secondary notes

## A. "Base Models Look Human To AI Detectors" (Xu, Zhong, Raghunathan, Fang, Kolter; CMU; arXiv 2605.19516v1)
Read: main body (pp. 1-10) + App. A.1-A.3. NOT opened: chunks 006-013 (A.4 output-layer-only adaptation and
App. B qualitative examples).

**Claim:** commercial detectors rate base-model continuations as overwhelmingly human, and instruct-model
continuations as AI, under identical prefixes. Detectors "are tracking artifacts of instruction tuning and local
context more than any invariant notion of machine-generated text" (abstract, S5).

**Detectors:** GPTZero and Pangram only (commercial APIs, research access). Metric = probability assigned to the
human label. No open-source detector is evaluated (Binoculars/DetectGPT cited only as related work).

**Continuation evaluation (S1, S4.1, A.1) -- the directly relevant experiment**
- 256 human-written + 256 AI-generated passages from RAID (abstracts, books, news, wiki) and MAGE (xsum, cnn,
  tldr, squad + their human counterparts). Prefix = first sentence only; one continuation per prefix; only the
  generated text is scored. Generation: temperature 1.0, top-p 0.95, vLLM.
- Models: Llama-3-8B and Qwen3-8B, base vs instruct.
- Llama-3-8B, human prefixes (exact, S1): base GPTZero 96.7% / Pangram 98.8% human;
  instruct GPTZero 30.3% / Pangram 17.1% human.
- Other cells only in Figs. 1/6 (approximate bar readings, mine): Llama base, AI prefix ~0.93-0.95; Llama instruct,
  AI prefix ~0.30 GPTZero / ~0.12 Pangram; Qwen3 base ~0.93 / 0.83 (GPTZero human/AI prefix), ~0.88 / 0.77
  (Pangram); Qwen3 instruct ~0.55 / 0.50 (GPTZero), ~0.28 / 0.20 (Pangram).
- Human prefix > AI prefix by a small margin in every cell ("human context" effect).
- How the instruct model was prompted for continuation (chat template or raw text) is not stated in what I read.
- Controls (A.1, Fig. 6): "Instruct + HIP" used as a plain generator and "Instruct + FT" (instruct model
  further trained on FineWeb-Edu with plain LM loss) both recover most of the human-likeness (Llama ~0.8-0.97).
  "Detector judgments are sensitive to post-training state itself."

**HIP (S3-4), secondary for us:** LoRA (rank 128, alpha 128, dropout 0.05, lr 5e-5, 1 epoch, 11,757 pairs of
GPT-5-nano paraphrase -> original human text, plain `<source_text>/<target_text>` format, no chat template),
applied for N=10 rounds. Qwen3 0.6B/1.7B/4B/8B/14B and Llama3 8B/70B, base and instruct. Human probability rises
each round while semantic score (GPT-5-nano, 0-10) falls. Eval set: 256 AI passages (32 x 8 sources).
Beats Simple Paraphrase, DIPPER, SilverSpeak, StealthRL on both detectors (Fig. 4). Works on instruct checkpoints
too and with native chat templates (A.3). Fails on GPT-4.1-nano via the OpenAI fine-tuning API (A.2).

**Limitations:** detectors will be updated; only two detectors; no mechanistic/activation analysis; detector
cost at list price ~$6000 GPTZero (40M words) + ~$5000 Pangram (100k passages).

**Implications for our project**
- "Sounds like AI" to detectors is largely "sounds like a post-trained assistant". A human-vs-AI probe trained
  on instruct-model outputs may therefore largely be an instruct-vs-base / assistant-register direction; the
  Assistant Axis comparison is the right control.
- Base-vs-instruct activation difference on matched prefixes is a candidate third direction to compare.
- Prefix provenance shifts scores, so hold prompts fixed across steering conditions.
- Their detectors are paid APIs; an open detector's base-vs-instruct gap is not established by this paper.
- Datasets worth reusing: RAID and MAGE subsets above (same-domain human and AI text).
- Code: github.com/YixuanEvenXu/humanization-by-iterative-paraphrasing; data + LoRA adapters in HF collection
  YixuanEvenXu/humanization-by-iterative-paraphrasing.

## B. "LLM Self-Recognition: Steering and Retrieving Activation Signatures" (Ardoin, Schaefer, Wunder; FU Berlin; ICML 2026; arXiv 2606.06315v1)
Read: main body (pp. 1-8) + App. A-F. NOT opened: chunk 006 (App. G cross-architecture evaluation).

**Self-recognition probe (S2.4.1, S3.1)**
- Models: Llama-3.1-8B, Llama-3.2-3B, Llama-3.2-1B, Ministral-3-8B, instruct variants.
- Feature: mean over tokens of the layer-l activation ("fixed, consistent location within each block";
  exact hook point not specified). Classifier: LDA with Ledoit-Wolf shrinkage on standardised features, 80/20
  split. Layer chosen per model near the middle.
- Data: 8,192 XL-Sum English BBC articles (<= 2,048 chars) with human summaries vs the model's own 1-2 sentence
  summaries (10-25 words requested; prompts in App. D). Sampling: top-p 0.9, temperature 0.7.
- AUROC, activation probe vs own-perplexity baseline (Table 1):
  with prompt: Ministral 100 / 99.71; Llama-8B 99.99 / 99.19; 3B 99.96 / 99.43; 1B 99.82 / 97.07.
  text only: Ministral 99.99 / 32.33; Llama-8B 99.16 / 47.86; 3B 99.03 / 47.49; 1B 98.58 / 52.27.
  Linear probe on text alone is near-perfect even at 1B; perplexity alone is at chance without the prompt.
- Layers (Table 4, App. E): best 12/15 of 32 (8B), 10/14 of 28 (3B), 6/8 of 16 (1B), 13 of 34 (Ministral);
  layer 0 worst; excluding it, spread <= 0.56 AUROC pts (with prompt), <= 1.2 (no prompt). Signal is spread across depth.
- Confound checks (App. E): cross-news-domain AUROC >= 99.49 (Table 5); lowercasing + punctuation stripping +
  length matching (closest 10% of pairs, < 22 chars apart) changes accuracy by < 0.5 pp (Table 6). Lengths:
  human 25.7 tokens (SD 8.85) vs 27.1-31.0 (Llama), 44.0 (Ministral).
- The probe is only tested on the same model's own outputs vs humans, in one genre (news summaries).

**Steering as watermark (S2.3, S3.2-3.5)** -- note this is NOT steering along the human-vs-AI direction
- Add alpha*v at one middle layer at every generated token; v is a RANDOM sparse vector, U([-1,1]^d) with 99.7%
  of entries zeroed; alpha = 5; repetition penalty 1.1; same layer for injection and read-out.
- 2-way attribution F1 (Table 2, MLP 2x32 on token activations, majority vote): ELI5 text-level 100 / 99.1 /
  95.5 / 85.3 and Fresh News 100 / 99.1 / 88.3 / 83.8 for Ministral / Llama-8B / 3B / 1B. Degrades with more
  classes (Fig. 2, ~0.7 text-level at 20).
- Training-free: cosine between the unsteered model's activations on the steered text and v is ~5e-3 (Fig. 5);
  accuracy 84.6% text-level, 77.8% after DIPPER-XXL paraphrase (lex 60, order 20) (Table 3). An injected residual
  direction survives sampling and re-encoding.
- Quality (App. F): NVIDIA quality-classifier-deberta score roughly unchanged; MMLU −1 to −2% relative for Llama,
  −5.5 to −6% for Ministral. Sparse vectors give a better detectability/quality trade-off than dense (Fig. 4).
- Discussion says detection falls to chance across architectures (App. G, not read).

**Limitations stated:** white-box only; activation pass costs ~21% of generation time; quality classifier
biased against markdown; some random vectors degrade quality more than others; other attacks untested.

**Implications for our project**
- Supports feasibility: a mean-pooled linear human-vs-AI direction exists in 1B-8B models at mid layers.
- Does not answer our question: they never steer along the learned direction, and never score steered text with
  an independent detector. Related-work pointers that do: Kuznetsov et al. 2503.03601 (SAE feature steering,
  in our papers/), Ackerman & Panickssery 2410.02064 (self-recognition vector steers authorship claims).
- Random sparse vectors at alpha=5 are a ready-made control: a norm-matched random direction changes
  activations detectably with little quality change, so our probe-direction effect must exceed that.
- Read-back check to copy: after steering, re-encode the output unsteered and measure projection on d.
- Code/data: github.com/Thibaud-Ardoin/LLM-Self-Recognition (includes Fresh News, Guardian Nov 2025-Jan 2026).
