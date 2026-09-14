# Verdict -> RESULTS_AMENDED

Primary cell: concept `side`, informative context, primary block, rank 1, direction fitted on `bn`. Readout `pair_acc`. Values are mean ± std over seeds, in accuracy points.

## bigscience/bloom-1b7 (block 12)

**FALSIFIED**

- F1 concept direction is no worse than random (specificity 1.3 pts)
- F2 cross-language gap 1.5 pts < 5.0 pts: shared encoding
- F3 matched concept shows the same gap (0.0 pts): language-channel artifact

| metric             | mean  | std  |
| ------------------ | ----- | ---- |
| clean_acc_src      | 0.96  | 0.05 |
| clean_acc_tgt      | 0.98  | 0.01 |
| clean_acc_identity | 0.42  | 0.13 |
| drop_src           | 1.17  | 1.25 |
| drop_tgt           | -0.33 | 1.55 |
| drop_random_src    | -0.17 | 0.82 |
| gap                | 1.50  | 0.97 |
| specificity        | 1.33  | 1.63 |
| drop_identity      | 1.00  | 3.09 |
| gap_matched        | 0.00  | 0.00 |

Language-identity control: not interpretable (hi clean accuracy 0.42 < gate 0.65).
Sensitivity, rank 4: gap 4.2 ± 2.5, random drop -0.7

## bigscience/bloom-560m (block 12)

**FALSIFIED**

- F1 concept direction is no worse than random (specificity 1.7 pts)
- F2 cross-language gap -0.0 pts < 5.0 pts: shared encoding
- F3 matched concept shows the same gap (0.0 pts): language-channel artifact

| metric             | mean  | std  |
| ------------------ | ----- | ---- |
| clean_acc_src      | 0.97  | 0.03 |
| clean_acc_tgt      | 0.98  | 0.02 |
| clean_acc_identity | 0.99  | 0.01 |
| drop_src           | 1.00  | 1.33 |
| drop_tgt           | 1.00  | 0.82 |
| drop_random_src    | -0.67 | 1.33 |
| gap                | -0.00 | 1.05 |
| specificity        | 1.67  | 2.58 |
| drop_identity      | 0.67  | 1.33 |
| gap_matched        | 0.00  | 0.00 |

Language-identity control: the hi drop is closer to the source language -> effect tracks the lexical distinction.
Sensitivity, rank 4: gap 1.7 ± 3.0, random drop -0.3

# Verdict -> RESULTS_BACKUP

Primary cell: concept `side_aunt`, informative context, primary block, rank 1, direction fitted on `bn`. Readout `pair_acc`. Values are mean ± std over seeds, in accuracy points.

## bigscience/bloom-1b7 (block 12)

**SUPPORTED**

- gap 16.8 pts >= 15.0 pts, random drop -0.8 pts <= 3.0 pts

| metric             | mean  | std  |
| ------------------ | ----- | ---- |
| clean_acc_src      | 0.93  | 0.08 |
| clean_acc_tgt      | 0.86  | 0.05 |
| clean_acc_identity | 0.52  | 0.07 |
| drop_src           | 21.17 | 8.33 |
| drop_tgt           | 4.33  | 4.03 |
| drop_random_src    | -0.83 | 0.53 |
| gap                | 16.83 | 8.98 |
| specificity        | 22.00 | 8.34 |
| drop_identity      | 2.67  | 0.97 |
| gap_matched        | 0.00  | 0.00 |

Language-identity control: not interpretable (hi clean accuracy 0.52 < gate 0.65).
Sensitivity, rank 4: gap 53.7 ± 16.1, random drop -0.2

## bigscience/bloom-560m (block 12)

**INCONCLUSIVE**

- task gate failed: clean accuracy src 1.00, tgt 0.63 (need >= 0.65)

| metric             | mean  | std  |
| ------------------ | ----- | ---- |
| clean_acc_src      | 1.00  | 0.00 |
| clean_acc_tgt      | 0.63  | 0.12 |
| clean_acc_identity | 1.00  | 0.00 |
| drop_src           | 0.00  | 0.00 |
| drop_tgt           | 0.67  | 3.09 |
| drop_random_src    | 0.00  | 0.00 |
| gap                | -0.67 | 3.09 |
| specificity        | 0.00  | 0.00 |
| drop_identity      | 0.00  | 0.00 |
| gap_matched        | 0.00  | 0.00 |

Language-identity control: the hi drop is closer to the source language -> effect tracks the lexical distinction.
Sensitivity, rank 4: gap 5.0 ± 1.8, random drop 0.0
