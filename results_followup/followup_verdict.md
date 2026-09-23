# Follow-up verdict

Config: `configs/followup.yaml`. Screen: `screen.json`.

| model | side_aunt/bn | side_aunt/en | side_aunt_named/bn | side_aunt_named/en | gender_maternal/bn | gender_maternal/en |
|---|---|---|---|---|---|---|
| bigscience/bloom-1b7 | 0.93 | 0.85 | 0.93 | 1.00 | 1.00 | 1.00 |
| Qwen/Qwen3-1.7B-Base | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| Qwen/Qwen2.5-1.5B | 0.51 | 1.00 | 0.52 | 1.00 | 1.00 | 1.00 |
| ai-forever/mGPT | 0.38 | 0.10 | 0.38 | 1.00 | 0.60 | 1.00 |

No-ablation pair accuracy, all names, informative context.
Selected second-family model: `None`. Selected matched control: `None`.

## Outcome: NOT TESTABLE

- no second-family candidate passed the task gate; no ablation was run.
