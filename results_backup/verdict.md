# Verdict

Primary cell: concept `side_aunt`, informative context, primary block, rank 1, direction fitted on `bn`. Readout `pair_acc`. Values are mean ± std over seeds, in accuracy points.

## bigscience/bloom-1b7 (block 12)

**SUPPORTED**

- gap 16.8 pts >= 15.0 pts, random drop -0.8 pts <= 3.0 pts

| metric | mean | std |
|---|---|---|
| clean_acc_src | 0.93 | 0.08 |
| clean_acc_tgt | 0.86 | 0.05 |
| clean_acc_identity | 0.52 | 0.07 |
| drop_src | 21.17 | 8.33 |
| drop_tgt | 4.33 | 4.03 |
| drop_random_src | -0.83 | 0.53 |
| gap | 16.83 | 8.98 |
| specificity | 22.00 | 8.34 |
| drop_identity | 2.67 | 0.97 |
| gap_matched | 0.00 | 0.00 |

Language-identity control: not interpretable (hi clean accuracy 0.52 < gate 0.65).
Sensitivity, rank 4: gap 53.7 ± 16.1, random drop -0.2

## bigscience/bloom-560m (block 12)

**INCONCLUSIVE**

- task gate failed: clean accuracy src 1.00, tgt 0.63 (need >= 0.65)

| metric | mean | std |
|---|---|---|
| clean_acc_src | 1.00 | 0.00 |
| clean_acc_tgt | 0.63 | 0.12 |
| clean_acc_identity | 1.00 | 0.00 |
| drop_src | 0.00 | 0.00 |
| drop_tgt | 0.67 | 3.09 |
| drop_random_src | 0.00 | 0.00 |
| gap | -0.67 | 3.09 |
| specificity | 0.00 | 0.00 |
| drop_identity | 0.00 | 0.00 |
| gap_matched | 0.00 | 0.00 |

Language-identity control: the hi drop is closer to the source language -> effect tracks the lexical distinction.
Sensitivity, rank 4: gap 5.0 ± 1.8, random drop 0.0

