# Pilot run: seed 0

Date: 14 Sep 2026 · Machine: RTX 3070 (8 GB), Windows, `.venv` Python · Command: `python -X faulthandler src/run_all.py --seed 0` · Config: `configs/default.yaml` (unchanged, thresholds as pre-registered)

**Status: pilot only.** One seed, so the std columns are 0 by construction. These numbers are not the result of the experiment and must not be quoted as one.

## Verdict (as produced by `analyze.py`)

| model | verdict | reason |
|---|---|---|
| facebook/xglm-564M | **INCONCLUSIVE** | task gate failed: clean accuracy bn 0.40, en 0.50 (need ≥ 0.65 in both) |
| bigscience/bloom-560m | **INCONCLUSIVE** | task gate failed: clean accuracy bn 0.67, en 0.51 (need ≥ 0.65 in both) |

Neither model can answer the side question well enough, in both the source and target language, for ablation drops to mean anything. The pre-registered gate worked as intended.

## Primary cell

Concept `side`, informative context, block 12 of 24, rank 1, direction fitted on `bn`, 6 held-out names (120 items per language).
Clean accuracies are fractions. Everything else is in accuracy points (clean − ablated).

| metric | xglm-564M | bloom-560m |
|---|---|---|
| clean_acc_src (bn) | 0.40 | 0.67 |
| clean_acc_tgt (en) | 0.50 | 0.51 |
| drop_src (bn, v_bn) | −0.83 | 1.67 |
| drop_tgt (en, v_bn) | 0.00 | 0.00 |
| drop_random_src (bn, random) | 0.00 | −0.83 |
| **gap G** (primary metric) | **−0.83** | **1.67** |
| specificity | −0.83 | 2.50 |
| drop_identity (hi, v_bn) | 0.00 | 0.00 |
| gap_matched (gender) | −5.00 | 2.50 |
| rank-4 sensitivity: gap / random drop | 21.7 / 2.5 | 1.7 / 0.0 |

## Reading

1. **Models can't do the task.** English sits at chance (0.50 / 0.51) even though the context sentence states the side outright. XGLM is *below* chance on Bengali (0.40). This looks like an answer preference ("always father", or always the first option) rather than the model reading context or the kin term.
2. **The exactly-zero drops fit that.** Both models show drop_tgt = 0.00 and drop_identity = 0.00. If a model gives the same answer to every item, nudging one direction can't change its accuracy. This is a guess until the `logit_diff` column and per-label accuracy in `results.csv` confirm it.
3. **Ignore XGLM's rank-4 gap of 21.7 points.** It's a secondary cell, measured in a model that fails the gate, and one seed. Promoting it would be exactly the outcome-driven selection the spec rules out.
4. **The "language-identity control" line in `verdict.md` is meaningless here.** It compares drops that are all at or near zero. Disregard it while the gate fails.

## Runtime (measured, replaces README estimates)

| model | extraction (once) | one seed (probes + 104 ablations) |
|---|---|---|
| xglm-564M | 16 s | 463 s |
| bloom-560m | 16 s | 477 s |

Projected full run (5 seeds × 2 models) ≈ **80 min** on the 3070, plus weight downloads on first use (~1.1 GB per model).

## Next step (decide before running more seeds)

Running seeds 1–4 on these models can only give INCONCLUSIVE, so don't spend the time on it.

- **Pre-registered fallback (PRD §9):** move to `bigscience/bloom-1b7` and `facebook/xglm-1.7B`, keep everything else fixed, and rerun `--seed 0`. At float16 these fit in 8 GB. The config says float32, and float32 1.7B also fits for scoring at a reduced batch size (`--batch-size 8`).
- **Diagnose first (cheap, uses existing output):** in `results/results.csv`, filter `direction == none`, `concept == side`. Check accuracy and `logit_diff` for each language and context. If `logit_diff` has the same sign for both labels, the model has an answer bias, and a larger model may still fail. In that case the readout (question format) is the problem, not model size.
- If the readout gets changed (for example, a calibrated score comparing each item against its minimal-pair partner), it's a **deviation from the spec**. Log it here with the reason *before* rerunning, and state it in SPEC.md.

Minor: the `hf_xet not installed` warning is harmless (downloads fall back to HTTP). `pip install hf_xet` would speed up downloads.
