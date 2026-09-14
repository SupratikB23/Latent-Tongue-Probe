# latent-tongue-probe

A one-week experiment for The Bu1LD research screen, thread T-02 ("Sapir Whorf and the Latent Tongue"). It asks whether a multilingual LM encodes **paternal vs maternal uncle** as a direction specific to Bengali (কাকা/মামা), where the language forces the distinction, and not to English, where "uncle" collapses it.

- One-page spec for the reviewer: `SPEC.md`. Full design: `PRD.md`. Where the project came from: `conversation-log.md`.
- Contact: Ryan Gomez (The Bu1LD). Email and GitHub are the channels. Only send work that can be inspected; nothing from the unpublished CoT-Mediate results.

## Commands

```bash
python src/build_stimuli.py --preview                 # print minimal pairs for native-speaker review
python src/build_stimuli.py                           # regenerate data/stimuli.jsonl
python -m pytest tests -q                             # offline tests, CPU, ~1-2 min
python src/run_all.py --config configs/smoke.yaml     # offline end-to-end run on a tiny random model
python src/run_all.py --seed 0                        # real run, one seed (GPU)
python src/run_all.py                                 # original pre-registration, all 5 seeds
python src/run_all.py --config configs/amended.yaml   # reported design (readout pair_acc, BLOOM-560m + 1b7)
python src/run_all.py --config configs/backup.yaml    # backup concept side_aunt
python src/diagnose_baseline.py                       # answer bias / pair accuracy from cached no-ablation scores
python src/per_seed.py --config configs/backup.yaml   # per-seed primary cell + matched-control ceiling check
python src/analyze.py                                 # re-aggregate results/raw -> results/
```

## Layout

`src/lexicon.py` holds every natural-language string (names, predicates, kin terms, templates). `build_stimuli.py` combines them. `extract.py` caches kin-token activations. `probe.py` covers probes, INLP subspaces, and erasure. `ablate.py` scores continuations under subspace mean-ablation. `run_all.py` is the entry point. `analyze.py` applies the decision rule and draws the figures. Config lives in `configs/default.yaml`.

## Compute

The main machine is a **separate PC with an RTX 3070 (8 GB)**. This laptop has no GPU; use it for editing, tests, and the smoke config only. Kaggle (T4/P100, 30 GPU-h/week) is the fallback. Reported models: BLOOM-560m (float32) and BLOOM-1b7 (bfloat16, the only way it fits in 8 GB). A full 5-seed config takes ~96 min. Setup and the unattended runner (`overnight.ps1`) are in `scripts/`.

## Status (14 Sep 2026)

The run is complete. Results are in `results/final_results.md`, and the amendment log is `results/pilot_seed0.md`. Uncle side was FALSIFIED in both BLOOM models. The backup aunt side was SUPPORTED in BLOOM-1b7 only (G 16.8 ± 9.0), with the matched control at ceiling. Lead with the falsification. Never present the backup as the headline. `configs/default.yaml` is the untouched pre-registration. New changes go in a new config plus a log entry.

On the GPU box, activate `.venv` first. The system Python there crashes on sentencepiece. BLOOM-560m must run in float32.

## Rules

- **Scope is frozen.** One concept (side), one matched control concept (gender), two models, three languages, one primary metric, five seeds. Cut before adding. `side_aunt` is a backup and only runs if `side` turns out flat.
- **Thresholds in `configs/default.yaml` are pre-registered.** Never change them after seeing results. Report the verdict `analyze.py` produces, including FALSIFIED or INCONCLUSIVE.
- The primary cell is block `n_layers // 2`, rank 1, informative context, direction fitted on `bn`. The layer sweep and rank 4 are secondary. Never promote the best-looking cell to primary.
- The train/test split is name-disjoint, so a minimal pair never straddles it. Keep it that way.
- Any change to `lexicon.py` requires a native Bengali/Hindi review (`--preview`) and a rebuilt `data/stimuli.jsonl`. Keep text NFC-normalized.
- Commit raw per-seed CSVs (`results/raw/`), not just summaries. Never commit `cache/`.
- Be honest about the gap: the prior paper (CoT-Mediate) is behavioral, and this project is the first internals work. Don't claim fluency that doesn't exist yet.
- Code style: flat `src/` modules imported by name, type hints, short docstrings, no framework beyond torch/transformers/sklearn.
