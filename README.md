# Is there a "sounds like AI" direction in the residual stream?

Tests whether the linear direction that separates human-written from AI-written text in
Llama-3.1-8B(-Instruct) activations is causal for generation, and whether it is distinct from
formality, fluency, length, domain and the assistant persona. Full write-up: [REPORT.md](REPORT.md).

## Key findings

- **There is a steerable direction, and it is chat-model register, not AI authorship.** It
  separates chat-model text from human text at AUROC ≥ 0.97 across HAP-E, HC3 and RAID, but
  base-LLM text only at 0.75–0.78 and RAID non-chat generators at 0.64.
- **Causal for how the text reads, not for detection in general.** At matched coherence,
  steering toward the human side moves an LLM judge's AI-likelihood from 92.4 to 70.6 (texts
  flagged: 98.6% → 41%) and the desklib detector's flag rate from 99.5% to 85%. Equal-norm
  random directions, a formality direction, an assistant-axis proxy and a "write like a human"
  prompt do not. A Binoculars-style detector moves the opposite way and RADAR does not respond.
- **Not distinct from chat register.** Cosine 0.96 with "instruct-model text minus base-model
  text" (0.88 on disjoint generators); removing that component abolishes the steering effect.
  Cosine with formality 0.15, fluency 0.32, assistant-axis proxy −0.01.
- **Present in the base model; chat tuning moves the writing, not the representation.** The
  base and instruct directions have cosine 0.84. The model's own writing sits 0.45 SD from
  human text (base), 1.39 SD (instruct, raw continuation) and 3.9 SD (instruct, chat template).
  Adding the direction to the base model raises judged AI-likelihood from 59 to 92.5 while
  judged coherence goes up.
- **Costs.** Steering toward human lowers coherence (94.6 → 86.1), adds repetition and changes
  content; it is not a practical detector-evasion tool at preserved quality.

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
python datasets/download_datasets.py            # datasets are not in git
export PYTHONPATH=src HF_TOKEN=... OPENROUTER_KEY=...

python src/prep_data.py                         # corpora and prompts -> results/data/
python src/extract.py instruct && python src/extract.py base      # ~20 min each, 25 GB in results/acts/
python src/fit_acts.py instruct && python src/fit_acts.py base    # own-writing activations, assistant-axis proxy
python src/directions.py instruct && python src/directions.py base  # readout + all directions
python src/geometry.py                          # disjoint-data cosines, base vs instruct
python src/steer.py --plan pilot --split dev --out pilot_dev      # layer / coefficient choice
python src/steer.py --model instruct --split test --plan main --layer 12 --out main_test
python src/steer.py --model base --split cont_test --plan base --layer 12 --coefs 0.25,0.5,0.75 --out base_cont
python src/steer.py --model instruct --split cont_test --plan rawsmall --layer 12 --coefs 0.25,0.5,0.75 --out instruct_cont
python src/score.py results/gens/{refs,main_test,base_cont,instruct_cont}.jsonl   # detectors
for f in refs main_test base_cont instruct_cont; do python src/judge.py results/gens/$f.jsonl; done
python src/readback.py instruct main_test,instruct_cont,refs 12 && python src/readback.py base base_cont 12
python src/analyze_steer.py main_test test && python src/analyze_steer.py base_cont cont_test \
  && python src/analyze_steer.py instruct_cont cont_test
python src/summarize.py && python src/figures.py
```

Needs one 48 GB GPU (two 7B models are loaded together for the Binoculars score). About
5 hours end to end. LLM-judge calls: about 20k (roughly $40 at list prices).

## Files

| Path | Content |
|---|---|
| `REPORT.md` | Full report with tables and figures |
| `planning.md` | Pre-specified plan and hypotheses |
| `src/` | `prep_data`, `trunc_control`, `common` (hooks, pooling, generation), `extract`, `fit_acts`, `directions`, `geometry`, `steer`, `score`, `judge`, `readback`, `analyze_steer`, `summarize`, `figures` |
| `results/readout_*.json`, `results/geometry.json` | Readout AUROCs, cosines, projection-out, base vs instruct |
| `results/directions_*.npz` | All directions, per layer |
| `results/gens/*.jsonl` | Every generated text, with condition and prompt id |
| `results/scores/`, `results/judge/` | Detector scores, judge scores (and cache) per text |
| `results/table_*.csv`, `results/vs_random_*.csv`, `results/analysis_*.json` | Per-condition tables, tests against random controls, matched-coherence comparison |
| `results/truncation_control.csv`, `results/per_task_main_test.csv` | Length control, per-task breakdown |
| `figures/` | Five figures used in the report |
| `literature_review.md`, `resources.md`, `papers/`, `datasets/`, `code/` | Pre-gathered resources |
