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

---

## Diagnosis (14 Sep 2026, `src/diagnose_baseline.py`, all 20 names, no ablation)

`pick0` = share of items where answer 0 wins. `pair_acc` = minimal-pair accuracy (a constant answer preference cancels out; 0.50 = chance).

| model | concept / context | bn acc · pick0 · pair | en acc · pick0 · pair | hi acc · pick0 · pair |
|---|---|---|---|---|
| bloom-560m | side / informative | 0.56 · 0.90 · **0.97** | 0.55 · 0.81 · **0.98** | 0.61 · 0.27 · 0.99 |
| bloom-560m | side / neutral | 0.47 · 0.96 · **0.12** | 0.50 · 0.91 · 0.50 | 0.50 · 0.55 · 0.47 |
| xglm-564M | side / informative | 0.43 · 0.07 · **0.00** | 0.50 · 1.00 · **0.93** | 0.54 · 0.74 · 0.76 |
| xglm-564M | side / neutral | 0.50 · 0.00 · **0.06** | 0.50 · 1.00 · 0.50 | 0.50 · 0.95 · 0.99 |
| bloom-560m | gender / informative | 0.50 · 1.00 · 1.00 | 0.87 · 0.51 · 1.00 | 0.59 · 0.87 · 1.00 |
| xglm-564M | gender / informative | 0.50 · 1.00 · 0.90 | 0.68 · 0.82 · 1.00 | 0.50 · 1.00 · 1.00 |

Findings:
1. **Answer bias is confirmed.** `pick0` is near 0 or 1 in most cells. Item accuracy mostly measures which answer the model prefers.
2. **Pair accuracy recovers the signal where the model uses context.** On BLOOM side/informative it passes the 0.65 gate in bn (0.97) and en (0.98). XGLM passes in en (0.93) but not in bn (0.00).
3. **Bengali side is systematically reversed.** Neutral context gives 0.12 (BLOOM) and 0.06 (XGLM), and XGLM informative gives 0.00. The মামা item leans toward "বাবার" more than the কাকা item does. That is a consistent effect, not noise, but both models flip it and Hindi doesn't (XGLM hi neutral 0.99). **Unexplained.** Before trusting any Bengali number, check whether it's a stimulus/tokenization artifact (e.g. surface overlap between মামা and মায়ের, or the answer word in the question) or genuine model behavior.
4. **The probe figure is saturated and partly hidden.** Informative-context probes sit at 1.0 at nearly every block, in English too (the English "uncle" token carries the context side from block 0). Bengali curves were drawn underneath the Hindi/English lines. Fixed in `analyze.py` (line widths now differ). At 1.0, decodability can't separate the hypotheses; the transfer rows in `probe.csv` (train bn → test en) are the informative ones.

Code changes (no threshold changed):
- `results.csv` now also records `pair_acc` per row. The verdict still uses item `acc` until a deviation is logged below.
- `run_all.py --baseline-only` caches activations + un-ablated scores and stops, for cheap checks of new models.

**Proposed deviation (not yet adopted):** switch the gate and all drops from item accuracy to pair accuracy, and move to the pre-registered 1.7B fallback models. Adopt only after (a) the Bengali reversal in finding 3 is explained and (b) `--baseline-only` on the 1.7B models shows pair_acc ≥ 0.65 for side/informative in bn and en.
