# T-02 probe: Is "which side of the family" a Bengali-only direction?

Supratik Bhowal · spec for The Bu1LD research screen · repo: `latent-tongue-probe`

**Falsifiable hypothesis.** Take XGLM-564M and BLOOM-560m at decoder block 12/24. Fit the rank-1 direction that separates Bengali কাকা (father's brother) from মামা (mother's brother), then mean-ablate it. This lowers accuracy on a "father's or mother's brother?" question by ≥15 points more in Bengali than in English, where "uncle" gets its side only from a context sentence. A random rank-1 direction lowers Bengali accuracy by ≤3 points.

**Intervention.** Fit a logistic probe on the block-12 residual stream at the kin-term token (14 train names). Its weight vector, mapped to raw space, gives direction v. Then, at the output of block 12, at every token except the first:
h ← h − ((h − μ_lang)·v)v, where μ_lang is that language's label-free mean activation.
Readout: item is correct iff log p(correct answer) > log p(other answer), e.g. " father" vs " mother", " বাবার" vs " মায়ের".

**Model and dataset.** `facebook/xglm-564M` and `bigscience/bloom-560m`, both pretrained on Bengali and Hindi. Hand-built stimuli: *context sentence · kin-term sentence · question*, with 20 names × 10 neutral predicates × 2 labels. That makes **200 minimal pairs per cell**, across 3 languages (bn, hi, en), 2 concepts, and 2 context types: 4,800 items. The split is name-disjoint (14/6). Compute: one RTX 3070 (8 GB) or a Kaggle T4, budgeted under 4 GPU-hours end to end.

**Control conditions.** Each changes one thing and holds the rest fixed, the same logic as the provenance control in CoT-Mediate.
1. *Random direction* of equal rank, same hook, same μ.
2. *Matched concept.* Gender of the paternal sibling (কাকা/পিসি, चाचा/बुआ, uncle/aunt) runs through the identical pipeline. It is lexicalized in all three languages, so its cross-language gap should be ~0.
3. *Language identity.* Hindi lexicalizes side too (चाचा/मामा) in a different script, so its drop should track Bengali, not English.
4. *Sanity checks.* A neutral context ("Rahul has a big family.") should put English at chance but not Bengali. A shuffled-label probe gives the probe baseline.

**Primary metric (fixed in advance).** G = drop_bn − drop_en under v_bn, in accuracy points, at block 12, rank 1, mean ± std over 5 seeds. drop = clean accuracy − ablated accuracy. Support needs G ≥ 15 and random-direction drop ≤ 3. The result is only interpretable if clean accuracy ≥ 0.65 in bn and en; otherwise the verdict is inconclusive, not positive. The layer sweep and rank-4 runs are reported but cannot override the primary verdict.

**Falsifying results.**
- F1: The Bengali direction's drop is within 3 points of the random direction's. The effect is not concept-specific.
- F2: G < 5 points. Side is encoded in a shared multilingual subspace, and the Whorfian claim fails for this concept in these models.
- F3: The matched gender concept shows a gap within 5 points of G. I'm measuring a Bengali-vs-English channel artifact, not kinship.

A negative result is still a finding. "Kinship side lives in a shared subspace" or "decodable but not used" are both claims about multilingual representation, and I'll report whichever the code returns.

**Smallest reproducible artifact (one week).**
```
latent-tongue-probe/
  README.md               one command, expected runtime, expected sanity values
  requirements.txt        pinned (torch 2.7.0, transformers 4.57.3, scikit-learn 1.6.0)
  configs/default.yaml    models, seeds, pre-registered thresholds
  data/stimuli.jsonl      7,200 items; src/lexicon.py is the single reviewable source
  src/run_all.py          python src/run_all.py --seed 0
  src/{extract,probe,ablate,analyze}.py
  tests/                  offline: ablation invariant, scorer = full logits, stimulus invariants
  results/raw/            per-seed CSVs (committed) → results.csv, verdict.md, figure1.png
```
The decision rule is code (`analyze.decide`), so the verdict can't be tuned by hand after the fact.
