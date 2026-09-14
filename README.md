# latent-tongue-probe

Does a multilingual language model encode **paternal vs maternal uncle** as a direction specific to languages that lexicalize it (Bengali কাকা/মামা, Hindi चाचा/मामा), or as one shared across languages, including English, where "uncle" collapses the distinction?

This repo is a probe for The Bu1LD thread T-02. The one-page spec is [SPEC.md](SPEC.md) and the full design is [PRD.md](PRD.md).

## Reproduce

```bash
python src/run_all.py --config configs/amended.yaml    # reported design (Amendments 1-2) -> results_amended/
python src/run_all.py --config configs/backup.yaml     # pre-specified backup concept   -> results_backup/
python src/run_all.py                                  # original pre-registration      -> results/
```

On Windows, `powershell -ExecutionPolicy Bypass -File scripts\overnight.ps1` runs the first two unattended and logs to `logs/`. **Results: [results/final_results.md](results/final_results.md).** Amendment log: [results/pilot_seed0.md](results/pilot_seed0.md).

Outputs land in `results/`:

| file | content |
|---|---|
| `verdict.md` | SUPPORTED / FALSIFIED / INCONCLUSIVE per model, from the pre-registered rule in `configs/default.yaml` |
| `figure1.png` | accuracy drop in bn / hi / en under the Bengali side direction, a random direction, and the matched gender direction |
| `figure2_probe_layers.png` | probe accuracy by block, per language and context, with a shuffled-label control |
| `results.csv` | every ablation condition × seed × language × context (raw, long format) |
| `probe.csv`, `erasure.csv`, `summary.csv`, `gap_by_layer.csv` | secondary tables |
| `raw/<model>/seed<k>/` | per-seed CSVs + `run_meta.json` (GPU, versions, wall time) |

## Setup

Use Python 3.11 or 3.12. Install torch for your hardware first, then the pinned requirements.

**RTX 3070 PC (primary).** Clone or copy the repo, then run from the repo root:

```bash
powershell -ExecutionPolicy Bypass -File scripts\setup_remote.ps1
```

```bash
bash scripts/setup_remote.sh
```

Each script creates `.venv`, installs `torch==2.7.0` (CUDA 12.6 wheels) and `requirements.txt`, checks that the GPU is visible, and runs the tests. Then activate the venv and run `python src/run_all.py`. Model weights (~2.2 GB each) download from the Hugging Face Hub on first use.

**Kaggle (fallback).** Create a GPU notebook with internet on, then:

```bash
git clone <repo-url> && cd latent-tongue-probe
grep -v '^torch==' requirements.txt > req-kaggle.txt && pip install -r req-kaggle.txt
python src/run_all.py
```

**CPU only (no GPU).** Tests and the offline smoke pipeline:

```bash
pip install -r requirements.txt
python -m pytest tests -q
python src/run_all.py --config configs/smoke.yaml
```

The smoke config uses a 4-layer random GPT-2 with a tokenizer trained on the stimuli, so nothing is downloaded. Its numbers mean nothing; it only exercises the plumbing, writing to `results_smoke/`.

## Expected runtime

Measured on one RTX 3070 (8 GB), 5 seeds:

| run | per seed | total |
|---|---|---|
| `amended.yaml`: BLOOM-560m (float32) + BLOOM-1b7 (bfloat16) | ~8 + ~10.5 min | 96 min |
| `backup.yaml`: same models, aunt concept | same | 98 min |
| CPU smoke config | | a few minutes |

Kaggle T4 is untested; expect it to be slower.

If memory runs out, pass `--batch-size 16`. If time runs short, raise `ablation.layer_stride` in the config; the primary cell (block 12) is always included.

## Expected sanity values

These are the values that indicate the pipeline is working. They are not the result.

| check | where | expected |
|---|---|---|
| shuffled-label probe | `probe.csv`, `control == shuffled` | ≈ 0.5 at every block |
| bn / hi side probe, neutral context | `figure2` dashed lines | high (the kin term itself carries side) |
| en side probe, neutral context | `figure2` dashed English line | ≈ 0.5 (nothing to decode) |
| en side question, neutral context, no ablation | `results.csv`, `direction == none` | ≈ 0.5 |
| random-direction drop | `figure1` middle bars | ≈ 0 points |

If clean accuracy on the informative-context side question is below 0.65 in bn or en, the verdict is INCONCLUSIVE by design: the model can't do the task, so ablation drops mean nothing.

## Stimuli

`data/stimuli.jsonl` holds 7,200 items: 3 concepts × 2 context types × 3 languages × 20 names × 10 predicates × 2 labels. Each item has the form *context sentence · sentence with the kin term · question*:

```
রাহুলের মায়ের একজন ভাই আছেন। রাহুলের মামা কলকাতায় থাকেন। প্রশ্ন: রাহুলের মামা কি বাবার ভাই না মায়ের ভাই? উত্তর:   → " মায়ের"
Rahul's mother has one brother. Rahul's uncle lives in Kolkata. Question: Is Rahul's uncle the father's brother or the mother's brother? Answer:   → " mother"
```

Every string lives in [src/lexicon.py](src/lexicon.py). To review or edit:

```bash
python src/build_stimuli.py --preview   # one minimal pair per cell
python src/build_stimuli.py             # rebuild data/stimuli.jsonl after edits
python -m pytest tests/test_stimuli.py -q
```

## Layout

```
configs/default.yaml     models, seeds, pre-registered thresholds
configs/smoke.yaml       offline tiny-model pipeline check
data/stimuli.jsonl       generated stimuli (committed)
src/lexicon.py           all natural-language strings
src/build_stimuli.py     renders items, char spans, minimal-pair ids
src/models.py            loading, block lookup, tokenization helpers
src/extract.py           kin-token activations + per-language means (cached)
src/probe.py             probes, INLP subspaces, random subspaces, erasure
src/ablate.py            subspace mean-ablation hook + continuation scoring
src/run_all.py           entry point
src/analyze.py           decision rule, tables, figures
tests/                   offline tests
scripts/setup_remote.*   GPU machine setup
```
