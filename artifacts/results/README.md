# CRISP reproduction results

`^` higher is better, `v` lower is better. Blank cells were not evaluated
(generation columns are scored by the local Qwen3 rater; `--no-judge` leaves them empty).

| Run | Method | WMDP acc v | In-domain MMLU ^ | MMLU ^ | Fluency ^ | Concept v | Overall ^ |
|---|---|---|---|---|---|---|---|
| gemma2-2b_bio_original | original | 55.42 | 62.11 | 45.61 | 1.49 | 0.02 | 4.66 |
| gemma2-2b_bio_crisp | crisp | 55.42 | 62.56 | 46.49 | 1.50 | 0.02 | 4.66 |
| gemma2-2b_cyber_original | original | 33.60 | 44.00 | 45.61 | 1.29 | 0.10 | 17.88 |
| gemma2-2b_cyber_crisp | crisp | 33.40 | 42.00 | 44.74 | 1.36 | 0.11 | 18.95 |
