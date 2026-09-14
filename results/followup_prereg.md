# Follow-up pre-registration: aunt side, second model family, non-ceiling control

Frozen 14 Sep 2026, **before any follow-up run**. This is the one bounded follow-up to the aunt-side lead in [final_results.md](final_results.md) §2.
Config: [`configs/followup.yaml`](../configs/followup.yaml). Code: [`src/followup.py`](../src/followup.py). Nothing below is chosen after the run; the code applies every rule.

**Reproduce (one command, GPU):**

```bash
python src/followup.py
```

Output: `results_followup/` holds `screen.json`, `followup_verdict.md`, and the usual `verdict.md`, figures, and `raw/` per-seed CSVs.

## What stays untouched

- The pilot: `configs/default.yaml`, `configs/amended.yaml`, `configs/backup.yaml`, `data/stimuli.jsonl` (a test checks it is byte-for-byte what `lexicon.py` builds), `results/final_results.md`, `results/lab_log.md`. The pilot state is tagged `pilot-v1`.
- The uncle-side result (FALSIFIED in both BLOOM models). It isn't re-run or re-analyzed.
- **The intervention is not retuned around the aunt result:**
  - mean ablation at block `n_layers // 2`
  - rank 1, direction fitted on Bengali, informative context
  - minimal-pair accuracy
  - same probe settings, 6 of 20 names held out
  - every threshold identical to `configs/default.yaml`

  Rank 4 (which gave the largest aunt gap in the pilot) is still reported only and can't decide.
- The follow-up writes only to `results_followup/` and `cache_followup/`.

## What is frozen for the follow-up

| | frozen choice | why |
|---|---|---|
| hypothesis | `side_aunt` (পিসি / মাসি vs aunt / aunt) | the lead being tested |
| comparison | **G = drop_bn − drop_en** under the Bengali-fitted direction, primary cell, mean over seeds. SUPPORTED needs G ≥ 15 pts and random-direction drop ≤ 3 pts (`analyze.decide`, unchanged) | same metric as the pilot |
| anchor model | `bigscience/bloom-1b7`, bfloat16 | the model that produced the lead |
| second model family | first of **Qwen/Qwen3-1.7B-Base → Qwen/Qwen2.5-1.5B → ai-forever/mGPT** that passes the task gate (rule 1). Qwen in bfloat16, mGPT in float32 | XGLM failed the gate at both sizes, so a single fixed pick risked another amendment |
| matched control | first of **`side_aunt_named` → `gender_maternal`** that is off ceiling in every follow-up model (rule 2) | gender sat at 1.00 in the pilot, so F3 couldn't fire |
| seeds | **5, 6, 7, 8, 9** | fresh name splits and random directions; the pilot used 0–4 |
| languages | Bengali and English only | Ryan asked for bn vs en; Hindi wasn't interpretable in BLOOM-1b7 |
| BLOOM-560m | not included | failed the English gate on the aunt concept |

### Control candidates

Both are lexicalized in Bengali and English, so a real Bengali-specific effect predicts a control gap near 0.

- **`side_aunt_named`**: the aunt items with English naming the side. The English kin term is "paternal aunt" / "maternal aunt" instead of "aunt"; every Bengali item is identical to `side_aunt`. It holds concept, context, question and answers fixed and changes only whether English names the distinction. If G reflects Bengali naming the side, the gap should shrink here. If it reflects a generic Bengali-vs-English ablation effect, it stays.
- **`gender_maternal`**: মামা / মাসি vs uncle / aunt (mother's brother vs mother's sister), with answers "brother" / "sister". It's the pilot's gender control moved to the maternal side, sharing মাসি with the primary concept.

## Stage 1: no-ablation screen (runs first, inside the same command)

Un-ablated minimal-pair accuracy over all 20 names (200 pairs per cell), informative context, from the cached baseline scores. No ablation happens in this stage.

1. **Model rule.** Candidates are screened in order. The first whose `side_aunt` accuracy is ≥ 0.65 in both bn and en is the second model; later candidates are not screened.
2. **Control rule.** The first control candidate whose accuracy lies in **[0.65, 0.95]** in bn and en, in both BLOOM-1b7 and the selected model, is the matched control.
3. If either rule selects nothing, the outcome is **NOT TESTABLE**, no ablation runs, and that is reported as the result.

## Stage 2: the unchanged pipeline

`run_all.run_model` runs on BLOOM-1b7 and the selected model with concepts `[side_aunt, <control>]`, then `analyze.run` with the control as the matched concept, so F3 uses it.

## Outcome rule (code: `followup.outcome`)

A model **survives** when both hold:
- `analyze.decide` returns SUPPORTED.
- The control is informative. Its own Bengali direction must lower Bengali accuracy by ≥ 3 points more than a random direction does. Otherwise F3 can't fire and passes vacuously, which is what happened in the pilot.

| outcome | condition | next step |
|---|---|---|
| **SURVIVED** | both models survive | call to discuss expanding T-02 |
| **NULL** | both models FALSIFIED | reported as is |
| **MIXED** | anything else (one model only, INCONCLUSIVE, uninformative control) | reported as is |
| **NOT TESTABLE** | no eligible model or control in stage 1 | reported as is |

Every outcome is retained exactly as the code prints it in `results_followup/followup_verdict.md`, with per-seed values from `python src/per_seed.py --config configs/followup.yaml`. Once this is committed, any deviation (for example `--batch-size` after an out-of-memory error) gets logged in this file below the line, with its reason, before results are read.

## Known limits, fixed in advance

- Two model families, one size each. A survived result still covers only these two checkpoints.
- The Bengali stimuli for both controls reuse the already-built templates. Native review: `python src/build_stimuli.py --followup --preview`.
- BLOOM-1b7 and the Qwen models run in bfloat16 (8 GB limit). The screen measures accuracy in that precision.
- Runtime is not measured yet. Estimate: 5 seeds × 2 models × 2 concepts × 2 languages ≈ 1.5–2.5 h on the RTX 3070, plus model downloads.

---

## Deviations log

(none)
