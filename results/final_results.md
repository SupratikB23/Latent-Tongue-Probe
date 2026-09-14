# Final results: latent-tongue-probe

Run: 14 Sep 2026, 03:35–06:49, one RTX 3070 (`scripts/overnight.ps1`). Every step exited 0.
Primary design: `configs/amended.yaml` → `results_amended/` (96 min). Backup concept: `configs/backup.yaml` → `results_backup/` (98 min).
Amendments and their reasons: `results/pilot_seed0.md`. Thresholds are unchanged from the pre-registration.

All numbers: primary cell (block 12 of 24, rank 1, informative context, direction fitted on Bengali), minimal-pair accuracy, **mean ± std over 5 seeds**, in accuracy points. Clean accuracies are fractions.

## 1. Primary hypothesis: uncle side (কাকা / মামা): **FALSIFIED in both models**

| | BLOOM-1b7 (bfloat16) | BLOOM-560m (float32) |
|---|---|---|
| clean bn / en / hi | 0.96 ± 0.05 / 0.98 ± 0.01 / 0.42 ± 0.13 | 0.97 ± 0.03 / 0.98 ± 0.02 / 0.99 ± 0.01 |
| drop bn (v_bn) | 1.17 ± 1.25 | 1.00 ± 1.33 |
| drop en (v_bn) | −0.33 ± 1.55 | 1.00 ± 0.82 |
| drop bn (random) | −0.17 ± 0.82 | −0.67 ± 1.33 |
| **gap G** | **1.50 ± 0.97** | **−0.00 ± 1.05** |
| specificity | 1.33 ± 1.63 | 1.67 ± 2.58 |
| rank-4: G / random drop | 4.2 ± 2.5 / −0.7 | 1.7 ± 3.0 / −0.3 |
| verdict | FALSIFIED (F1, F2) | FALSIFIED (F1, F2) |

Both models pass the task gate comfortably. Removing the Bengali-fitted uncle-side direction at block 12 changes accuracy by about 1 point in **every** language, the same as a random direction. That holds at rank 4 as well. The probe decodes the distinction perfectly at the kin token (`figure2_probe_layers.png`), but the model doesn't route its answer through that direction. This is the "decodable but not used" outcome listed in PRD §2. It is not evidence of a shared subspace, because the Bengali direction doesn't hurt Bengali either.

## 2. Backup concept: aunt side (পিসি / মাসি): **SUPPORTED in BLOOM-1b7 only**

Run under the pre-specified contingency (`side` flat → run `side_aunt`). Reported separately; it does not change §1.

| | BLOOM-1b7 (bfloat16) | BLOOM-560m (float32) |
|---|---|---|
| clean bn / en / hi | 0.93 ± 0.08 / 0.86 ± 0.05 / 0.52 ± 0.07 | 1.00 ± 0.00 / **0.63 ± 0.12** / 1.00 ± 0.00 |
| drop bn (v_bn) | **21.17 ± 8.33** | 0.00 ± 0.00 |
| drop en (v_bn) | 4.33 ± 4.03 | 0.67 ± 3.09 |
| drop bn (random) | −0.83 ± 0.53 | 0.00 ± 0.00 |
| **gap G** | **16.83 ± 8.98** | −0.67 ± 3.09 |
| specificity | 22.00 ± 8.34 | 0.00 ± 0.00 |
| rank-4: G / random drop | 53.7 ± 16.1 / −0.2 | 5.0 ± 1.8 / 0.0 |
| verdict | **SUPPORTED** (G ≥ 15, random ≤ 3) | INCONCLUSIVE (en gate 0.63 < 0.65) |

In BLOOM-1b7, removing the Bengali-fitted aunt-side direction costs Bengali about 21 points and English about 4. A random direction costs nothing. At rank 4 the asymmetry grows to about 54 points. BLOOM-560m can't be read: it fails the English gate, and nothing moves in Bengali.

## 3. What limits the positive result (state all of these)

1. **Seed variance is large.** G = 16.8 ± 9.0 clears the 15-point threshold on the mean, as pre-registered, but its std is more than half the mean. Some individual seeds are probably below 15. Check with `python src/per_seed.py --config configs/backup.yaml` and report the per-seed values.
2. **The matched-concept control sits at ceiling.** gap_matched is exactly 0.00 ± 0.00 in all four runs. Gender pair accuracy most likely never moves, before or after ablation (verify with `per_seed.py`). F3 therefore couldn't detect a language-channel artifact. That falsifier is uninformative in this study, not passed.
   *Partial substitute:* the uncle-side concept runs through the same model, the same languages and the same pipeline, and shows **no** gap. A generic "Bengali channel" ablation artifact should have shown up there too. This argues against a generic artifact, but it wasn't the pre-registered control.
3. **The Hindi language-identity control isn't interpretable for BLOOM-1b7** (Hindi fails the gate for both concepts).
4. **One model.** BLOOM-560m doesn't replicate it (gate failure), and XGLM was dropped at the gate. The claim is limited to BLOOM-1b7.
5. **Two concepts were tested, and the positive one was the backup.** Lead with §1. §2 is a contingent, single-model result.
6. **Precision:** BLOOM-1b7 ran in bfloat16 (8 GB limit). bfloat16 visibly damaged BLOOM-560m's English margins.
7. **The mechanism is open.** Why aunt and not uncle is unexplained. Surface overlap (মাসি / মামা both contain মা, like the answer মায়ের) can't explain it on its own, because it applies to both concepts.

## 4. Notes on the generated verdict text

- The F3 bullet in the FALSIFIED verdicts fires vacuously (both gaps ≈ 0, and gender at ceiling). F1 and F2 alone decide.
- BLOOM-560m prints "hi drop is closer to the source language → effect tracks the lexical distinction". That comparison is between drops of ~1 point and means nothing.

## 5. One-paragraph summary (for the reviewer)

The pre-registered hypothesis failed. In BLOOM-560m and BLOOM-1b7, a linear direction that perfectly separates কাকা from মামা can be removed at mid-depth without affecting Bengali or English answers (gap 1.5 ± 1.0 and 0.0 ± 1.1 points; random-direction baseline ≈ 0). The kinship-side distinction is decodable but not used through that direction. The pre-specified backup concept, পিসি vs মাসি, behaves differently in BLOOM-1b7. Removing its Bengali direction costs Bengali 21 ± 8 points and English 4 ± 4, clearing the pre-registered bar (gap 16.8 ± 9.0). It doesn't replicate in the smaller model, and the matched-concept control was at ceiling. So this is a lead worth one follow-up experiment, not a finding.
