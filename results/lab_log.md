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

**Proposed deviation (superseded by Amendment 1 below):** switch the gate and all drops from item accuracy to pair accuracy, and move to the pre-registered 1.7B fallback models. Adopt only after (a) the Bengali reversal in finding 3 is explained and (b) `--baseline-only` on the 1.7B models shows pair_acc ≥ 0.65 for side/informative in bn and en.

---

## 1.7B fallback diagnostics (14 Sep 2026, `--baseline-only --dtype bfloat16`)

| model | side/inf bn · en (pair) | side/neutral bn | side/inf hi · neutral hi | gender/inf bn · en |
|---|---|---|---|---|
| bloom-1b7 | **0.96 · 0.98** | **0.89** | 0.31 · 0.03 | 1.00 · 1.00 |
| xglm-1.7B | 0.08 · 0.47 | 0.24 | 0.47 · 0.47 | 0.30 · 0.61 |

Findings:
1. **BLOOM-1b7 passes the gate and is the first model that gets কাকা/মামা right from the word alone** (neutral bn 0.89, where BLOOM-560m gave 0.12).
2. **Answering from the kin word alone flips between languages across checkpoints.** BLOOM-560m reverses bn and is correct on hi. BLOOM-1b7 is correct on bn and reverses hi. XGLM-564M reverses bn and is correct on hi. The same stimuli give opposite signs in different models, which points away from a Bengali-specific stimulus bug and toward model-specific associations. A native review of the bn and hi wording is still required.
3. **XGLM fails at both sizes.** At 1.7B even English with context is at chance (0.47). It is dropped as a *task-gate failure*, not because of any ablation outcome, and this is reported as a result.

## Amendment 1 (adopted; `configs/amended.yaml`)

Decided from the no-ablation diagnostics above. No ablation result was used. `configs/default.yaml` stays untouched as the original pre-registration.

| | pre-registered (`default.yaml`) | amended (`amended.yaml`) | reason |
|---|---|---|---|
| readout | item accuracy | **minimal-pair accuracy** | constant answer bias (`pick0` ≈ 0 or 1) masks what the input changes |
| models | XGLM-564M, BLOOM-560m | **BLOOM-560m, BLOOM-1b7** | only BLOOM passes the gate; the fallback named in PRD §9 was the 1.7B size |
| dtype | float32 | bfloat16 | 1.7B must fit in 8 GB |
| thresholds, primary cell, seeds, controls | — | **unchanged** | |

Consequences, fixed now:
- Cross-architecture replication is lost. The model pair is a **scale pair within one family**. Claims are limited to BLOOM.
- The Hindi language-identity control is only interpretable where Hindi passes the gate. `analyze.py` now prints "not interpretable" otherwise, which is expected for BLOOM-1b7.
- The 560m-vs-1b7 difference in bare-word bn accuracy (0.12 vs 0.89) is reported descriptively. No new hypothesis is added for it.

---

## Amended run, seed 0 (14 Sep 2026, `configs/amended.yaml --seed 0`)

Runtime on the RTX 3070 (bfloat16): BLOOM-560m 258 s per seed, BLOOM-1b7 632 s per seed. Test set = 6 names, 60 pairs per cell, so pair_acc moves in steps of 1.7 points.

| metric (pts unless clean) | bloom-1b7 | bloom-560m |
|---|---|---|
| clean pair_acc bn / en / hi | 1.00 / 0.97 / 0.24 | 0.97 / **0.59** / 0.97 |
| drop bn (v_bn) | 0.83 | −1.67 |
| drop en (v_bn) | −1.67 | 5.83 |
| drop bn (random) | 0.83 | 1.67 |
| gap G | 2.50 | −7.50 |
| specificity | 0.00 | −3.33 |
| gap_matched (gender) | 0.00 | 8.33 |
| rank-4 gap / random drop | 3.3 / 0.0 | 3.3 / −1.7 |
| **verdict** | **FALSIFIED** (F1, F2, F3) | **INCONCLUSIVE** (gate: en 0.59) |

Reading, one seed, not final:

1. **BLOOM-1b7: ablating the Bengali side direction changes nothing, in any language.** Bengali itself drops 0.8 points, the same as a random direction. So the result is not "shared encoding" in the sense of the Bengali direction also hurting English. The Bengali direction doesn't hurt Bengali either. This is the "decodable but not used" outcome listed in PRD §2. The probe finds the direction at 1.0, but removing it at block 12 doesn't affect the answer. Rank 4 gives the same picture.
2. **F1 and F2 decide the verdict. F3 fires vacuously.** F3 checks gap_matched ≥ G − 5, which is automatically true when G ≈ 0. It is meant to catch an artifact when there *is* a gap. This is a weakness in the pre-registered rule, noted here and not "fixed", since the verdict doesn't depend on it.
3. **BLOOM-560m fails the gate in English (0.59), although the float32 diagnostic gave 0.98 on all names.** Likely cause: English margins are tiny for this model (item logit_diff ≈ 0.08), so bfloat16 rounding flips many pair comparisons. Not yet verified. See next steps.

Next steps (fixed before looking at anything else):
- Run `python src/diagnose_baseline.py` on the bfloat16 caches. If 560m en side/informative pair_acc falls well below 0.98 on all names, bfloat16 is the cause, and 560m returns to float32 (Amendment 2, a gate-only reason, logged before rerunning).
- BLOOM-1b7 passes the gate, so run its seeds 1–4 unchanged to get the pre-registered 5-seed verdict with error bars.
- The layer sweep (`gap_by_layer.csv`) may be described as exploratory in the write-up. It cannot change the verdict.

---

## bfloat16 check (14 Sep 2026, `diagnose_baseline.py` on the amended caches, all 20 names)

| cell (pair_acc) | bloom-560m float32 (earlier) | bloom-560m bfloat16 | bloom-1b7 bfloat16 |
|---|---|---|---|
| side / informative / bn | 0.97 | 0.98 | 0.96 |
| **side / informative / en** | **0.98** | **0.55** | 0.98 |
| side / neutral / bn | 0.12 | 0.17 | 0.89 |
| side / informative / hi | 0.99 | 0.96 | 0.32 |
| gender / informative / en | 1.00 | 0.75 | 1.00 |

**Confirmed: bfloat16 breaks BLOOM-560m in English** (side 0.98 → 0.55, gender 1.00 → 0.75). Bengali and Hindi barely move. The English margins in this model are small enough to be lost to bfloat16 rounding. BLOOM-1b7 is unchanged between its two bfloat16 caches and passes the gate comfortably. A float32 check of 1b7 isn't feasible in 8 GB, which is a stated limitation.

## Amendment 2 (adopted; `configs/amended.yaml`)

- BLOOM-560m returns to **float32** (`model_dtype`). BLOOM-1b7 stays bfloat16. The reason is gate-only: the no-ablation baseline changed, not an ablation result. The seed-0 BLOOM-560m verdict above (INCONCLUSIVE, en 0.59) is void and gets overwritten.
- `run_meta.json` now records the dtype of every run.

## Backup concept triggered (`configs/backup.yaml`)

`side` was flat in BLOOM-1b7 on seed 0 (specificity 0.0, G 2.5). The pre-specified rule (CLAUDE.md, PRD §4.1) says `side_aunt` (পিসি/মাসি, बुआ/मौसी, aunt/aunt) runs in that case. It uses the same models, readout, thresholds and seeds, with results in `results_backup/`. This decision depended on an outcome, but the contingency was written down before any run. The backup is reported separately and **cannot change the primary `side` verdict**.

## Unattended run

`scripts/overnight.ps1`: (1) amended 5 seeds, both models → (2) diagnostic table → (3) backup concept. Estimated ~3.5–4 h from the measured seed times (560m float32 ~8 min/seed, 1b7 ~10.5 min/seed, per concept config).
