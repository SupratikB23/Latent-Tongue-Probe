# Verdict

Primary cell: concept `side`, informative context, primary block, rank 1, direction fitted on `bn`. Values are mean ± std over seeds, in accuracy points.

## bigscience/bloom-560m (block 12)

**INCONCLUSIVE**

- task gate failed: clean accuracy src 0.67, tgt 0.51 (need >= 0.65)

| metric | mean | std |
|---|---|---|
| clean_acc_src | 0.67 | 0.00 |
| clean_acc_tgt | 0.51 | 0.00 |
| drop_src | 1.67 | 0.00 |
| drop_tgt | 0.00 | 0.00 |
| drop_random_src | -0.83 | 0.00 |
| gap | 1.67 | 0.00 |
| specificity | 2.50 | 0.00 |
| drop_identity | 0.00 | 0.00 |
| gap_matched | 2.50 | 0.00 |

Language-identity control: the hi drop is closer to the target language -> effect looks specific to one language.
Sensitivity, rank 4: gap 1.7 ± 0.0, random drop 0.0

## facebook/xglm-564M (block 12)

**INCONCLUSIVE**

- task gate failed: clean accuracy src 0.40, tgt 0.50 (need >= 0.65)

| metric | mean | std |
|---|---|---|
| clean_acc_src | 0.40 | 0.00 |
| clean_acc_tgt | 0.50 | 0.00 |
| drop_src | -0.83 | 0.00 |
| drop_tgt | 0.00 | 0.00 |
| drop_random_src | 0.00 | 0.00 |
| gap | -0.83 | 0.00 |
| specificity | -0.83 | 0.00 |
| drop_identity | 0.00 | 0.00 |
| gap_matched | -5.00 | 0.00 |

Language-identity control: the hi drop is closer to the target language -> effect looks specific to one language.
Sensitivity, rank 4: gap 21.7 ± 0.0, random drop 2.5

