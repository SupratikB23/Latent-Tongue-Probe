# Verdict

Primary cell: concept `side_aunt`, informative context, primary block, rank 1, direction fitted on `bn`. Readout `pair_acc`. Values are mean ± std over seeds, in accuracy points.

## __tiny_random__ (block 2)

**FALSIFIED**

- F2 cross-language gap -1.7 pts < 5.0 pts: shared encoding
- F3 matched concept shows the same gap (1.7 pts): language-channel artifact

| metric | mean | std |
|---|---|---|
| clean_acc_src | 1.00 | 0.00 |
| clean_acc_tgt | 1.00 | 0.00 |
| clean_acc_identity | 0.35 | 0.00 |
| drop_src | 21.67 | 0.00 |
| drop_tgt | 23.33 | 0.00 |
| drop_random_src | 0.00 | 0.00 |
| gap | -1.67 | 0.00 |
| specificity | 21.67 | 0.00 |
| drop_identity | 8.33 | 0.00 |
| gap_matched | 1.67 | 0.00 |

Language-identity control: not interpretable (hi clean accuracy 0.35 < gate 0.65).
Sensitivity, rank 2: gap 1.7 ± 0.0, random drop 11.7

