# Verdict

Primary cell: concept `side`, informative context, primary block, rank 1, direction fitted on `bn`. Readout `pair_acc`. Values are mean ± std over seeds, in accuracy points.

## bigscience/bloom-1b7 (block 12)

**FALSIFIED**

- F1 concept direction is no worse than random (specificity 1.3 pts)
- F2 cross-language gap 1.5 pts < 5.0 pts: shared encoding
- F3 matched concept shows the same gap (0.0 pts): language-channel artifact

| metric | mean | std |
|---|---|---|
| clean_acc_src | 0.96 | 0.05 |
| clean_acc_tgt | 0.98 | 0.01 |
| clean_acc_identity | 0.42 | 0.13 |
| drop_src | 1.17 | 1.25 |
| drop_tgt | -0.33 | 1.55 |
| drop_random_src | -0.17 | 0.82 |
| gap | 1.50 | 0.97 |
| specificity | 1.33 | 1.63 |
| drop_identity | 1.00 | 3.09 |
| gap_matched | 0.00 | 0.00 |

Language-identity control: not interpretable (hi clean accuracy 0.42 < gate 0.65).
Sensitivity, rank 4: gap 4.2 ± 2.5, random drop -0.7

## bigscience/bloom-560m (block 12)

**FALSIFIED**

- F1 concept direction is no worse than random (specificity 1.7 pts)
- F2 cross-language gap -0.0 pts < 5.0 pts: shared encoding
- F3 matched concept shows the same gap (0.0 pts): language-channel artifact

| metric | mean | std |
|---|---|---|
| clean_acc_src | 0.97 | 0.03 |
| clean_acc_tgt | 0.98 | 0.02 |
| clean_acc_identity | 0.99 | 0.01 |
| drop_src | 1.00 | 1.33 |
| drop_tgt | 1.00 | 0.82 |
| drop_random_src | -0.67 | 1.33 |
| gap | -0.00 | 1.05 |
| specificity | 1.67 | 2.58 |
| drop_identity | 0.67 | 1.33 |
| gap_matched | 0.00 | 0.00 |

Language-identity control: the hi drop is closer to the source language -> effect tracks the lexical distinction.
Sensitivity, rank 4: gap 1.7 ± 3.0, random drop -0.3

