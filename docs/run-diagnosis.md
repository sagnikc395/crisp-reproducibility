# Run diagnosis — 2026-08-07, A100-SXM4-40GB

Notes on the `gemma2-2b_bio` + `gemma2-2b_cyber` run (`--stages original,crisp`, judge on).

Short version: **the configuration is faithful to the paper and the pipeline is sound, but
the run does not reproduce the paper's result.** Two independent failures, plus a wall-clock
problem that is unrelated to either. Timings and numbers below are read off that run's log
and off `assets/2026.acl-long.82.pdf`, not estimated.

---

## 1. Where the 48 minutes went

`configs/gemma2-2b_bio.yaml` reported `exit code: 0 after 48 min`. Broken down:

| Phase | Wall clock | Notes |
| --- | --- | --- |
| `original`: weights + SAE download | ~1.5 min | one-off per session |
| `original`: MCQ eval (wmdp + retain + mmlu) | ~14 s | 1273 + 57 + 29 batches |
| `original`: 100 generations | 29 s | |
| **`original`: judge** | **17 min 32 s** | 11:26 fluency + 6:06 concept |
| `crisp`: SAE load (6 layers) | 19 s | cached after the first |
| `crisp`: corpus load (2 × 5000 docs) | **3 min 40 s** | read over the Drive symlink |
| `crisp`: feature selection | 17 s | cached; free on re-runs |
| `crisp`: **training** | **1 min 40 s** | 200 steps, the actual method |
| `crisp`: MCQ eval + generations | ~1 min | |
| **`crisp`: judge** | **19 min 44 s** | 13:16 fluency + 6:28 concept |

**The judge is 37 of the 48 minutes — 78%.** The thing being reproduced (feature selection
+ CRISP training + the accuracy numbers the paper's claim rests on) is under 5 minutes.

Cyber is worse: its judge runs at a steady 137 s/batch against bio's ~90 s average, so each
cyber stage spends ~30 min judging. Same ~4 minutes of real work underneath.

## 2. Why the judge costs that much

`Qwen/Qwen3-4B-Thinking-2507` scores 100 prefixes twice (fluency, then concept) = 200
ratings, and a *Thinking* checkpoint spends most of its `judge_max_new_tokens=2048` budget
reasoning before emitting the number that gets parsed. The `3/200 ratings unparsed` and
`4/200 ratings unparsed` warnings are the tell: those hit the 2048 cap mid-reasoning, so the
rest are using most of it.

The cost is ~200 × up-to-2048 decoded tokens from a 4B model — decode-bound and sequential.

`judge_batch_size: 16` already did its job; it is what took this from a ~100-minute judge
down to ~18 min per stage. It cannot go further: batching divides the *number of batches*,
not the tokens each sequence decodes, and 7 batches of 16 already saturates an A100's
bandwidth for a 4B model.

What actually reduces token count, by payoff:

1. **Drop the judge while iterating.** `--no-judge --skip-generation` costs the Fluency and
   Concept columns and nothing else; WMDP and both MMLU columns are unaffected. ~10x.
2. **Use a non-thinking rater** for the final table — the rating lands in the first few
   tokens instead of token ~1800.
3. **Fewer prefixes** — 100 → 30 is a 3x cut, and the means are stable well below 100.
4. **Lower `judge_max_new_tokens`** — cheapest to try, but with 3–4/200 already truncating,
   it trades wall clock for unparsed ratings.

## 3. The other avoidable cost: corpus loading over Drive

3 min 40 s for 10 000 documents is not CPU-bound; it is the 2.5 GB bio corpus read through
the Drive FUSE mount that `data/` symlinks to, paid on every training stage. Copying
`data/wmdp/*.jsonl` to local disk once per session and pointing `data.target_corpus` /
`data.retain_corpus` at the copies removes it.

---

## 4. Does the current configuration give the right answer? No.

### 4.1 The hyperparameters are faithful — this was checked against the paper

Appendix F (p. 2085–2100 of the PDF) specifies, for CRISP on Gemma-2-2B Bio: SAE layers
`[4, 6, 8, 10, 12, 14]`, finetuning layers `[3–9]`, k=30, λ=30, LoRA rank 8, learning rate
4×10⁻⁵, τ=3, β=0.99, γ=0.01, and — verbatim — *"define α as 1 − β"*.

`configs/gemma2-2b_bio.yaml` matches every one of those. **α=0.01 is the paper's own
setting, not a repo mistake.** An earlier note in this file argued the α default was
miscalibrated; that was wrong, and sweeping α is not the first thing to do.

### 4.2 But the result is nowhere near the paper's

Gemma-2-2B / WMDP-Bio, paper Table 1 versus this run:

| | Unlearn Acc ↓ | Retain Acc ↑ | MMLU ↑ | Fluency | Concept | Overall ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| paper, original | 55.26 | 55.27 | 46.30 | 1.07 | 1.78 | 54.37 |
| **ours, original** | **55.42** | 62.11 | 45.61 | 1.49 | **0.02** | **4.66** |
| paper, CRISP | **29.67** | 54.45 | 46.33 | 0.92 | 1.63 | 56.70 |
| **ours, CRISP** | **55.42** | 62.56 | 46.49 | 1.50 | **0.02** | **4.66** |

The original-model row reproduces well (55.42 vs 55.26 unlearn, 45.61 vs 46.30 MMLU), which
says the model, the MCQ harness and the data pipeline are all fine.

**The paper's CRISP drops WMDP-Bio from 55.26 to 29.67 — near the 25% chance floor. Ours
does not move at all.** That is a failed reproduction of the central claim, not a marginal
gap.

### 4.3 Why the training does nothing

The unlearning loss never falls. Across the twenty logged steps it oscillates between 2.26
and 3.81, averaging 3.12 over steps 10–100 and 2.82 over steps 110–200 — drift smaller than
its own step-to-step noise. After 200 steps the SAE features CRISP selected are activating
as much as they did at the start.

Substituting the logged step-200 values into `total_loss`:

| Term | Raw | × weight | Share |
| --- | --- | --- | --- |
| unlearn | 3.0975 | ×0.01 = 0.031 | 2% |
| retain | 1.3339 | ×0.99 = 1.321 | 65% |
| coherence | 68.52 | ×0.01 = 0.685 | 33% |

(0.031 + 1.321 + 0.685 = 2.037 = the logged `total 2.0368`.) With α=0.01 the unlearning
signal is deliberately small, so it needs *time* to accumulate against the two retention
terms. The paper never states how much time it gets:

> **Neither the training step count nor the batch size appears anywhere in the paper.**
> Searching the full text for "step", "epoch" and "batch" returns nothing in the methods or
> Appendix F.

This repo picked `steps: 200`, `batch_size: 2` — the model sees **400 target documents out
of the 5000 loaded**, in 100 seconds of training. That is the single largest unconstrained
degree of freedom between this implementation and the paper's, and a flat unlearning loss is
what under-training looks like. γ·coherence being 33% of the objective (and spiking to 91%
of a single step — step 120, coherence 1368.7) makes the same point: at 200 steps the update
is still dominated by noise from 20 curated sentences.

### 4.4 A second, independent failure: the Concept score

Our Concept is **0.02** where the paper's original model scores **1.78**. Concept measures
whether the target concept appears in the continuation (paper Table 10 prompt, 0–2), so an
un-unlearned Gemma should score *high*. Ours says the concept is absent in essentially all
100 continuations from the untouched model, which cannot be right.

It also poisons the headline column: `metrics.py:10` computes Overall as a harmonic mean
including `concept * 50`, and a harmonic mean with a ~0 term collapses. That is the whole
reason our Overall reads 4.66 against the paper's 54.37 — the number is not comparable, and
neither is any Overall-based comparison built on it.

(Related: `report.py` labels the column `Concept v` (lower is better), while the paper's
Table 1 has Concept ↑ and Eq. 12 treats it as higher-is-better. The label is wrong even once
the values are fixed.)

## 5. Checked and ruled out

- **The adapter is active at eval.** `cli.py:104` passes the same `PeftModel` it trained
  into `evaluate_model`; `model.py:84` disables LoRA only inside `_reference_acts` and
  restores on exit; `tests/test_integration.py:52` asserts adapter-disabled logits equal
  pre-LoRA logits *and* that a perturbed adapter's differ.
- **The hyperparameters match Appendix F** (§4.1).
- **The base model and MCQ harness are correct** — the original-model row reproduces the
  paper to within 0.2 points on unlearn accuracy and 0.7 on MMLU (§4.2).

## 6. What to run next

Feature selection is cached (keyed on `top_k`/`τ`), so a CRISP-only run without the judge is
~4 minutes. Sweep **training duration** first, since that is the unspecified parameter:

```bash
for S in 500 1000 2000; do
  python -m crisp train -c configs/gemma2-2b_bio.yaml \
    --run-name "gemma2-2b_bio_crisp_s${S}" -o "train.steps=${S}" \
    --no-judge --skip-generation
done
```

Roughly 4 + 8 + 16 minutes of training plus ~1 min of eval each. Each writes its own
`artifacts/results/<run-name>__test.json`, so `crisp report` picks them up as extra rows and
the headline `gemma2-2b_bio_crisp` run is untouched.

Watch the unlearning loss, not just the accuracy — compare the mean of the `unlearn` field
over the first and last quarter of `artifacts/runs/<name>/history.json`:

| Outcome | Reading |
| --- | --- |
| unlearn loss falls, WMDP drops toward 25% | reproduced; the missing ingredient was training duration, and that belongs in the write-up as a reproducibility gap in the paper |
| unlearn loss falls, WMDP stays at 55 | the suppressed features do not mediate the MCQ answer — a real negative result about the method |
| unlearn loss still flat at 2000 steps | the optimiser is not reducing the term at all; look at feature selection and the SAE encode path next, then α |

Separately, and independent of unlearning: fix the Concept scorer before quoting any Overall
number. Print a few raw judge outputs from the concept pass on the *original* model — if the
model is being asked correctly, most should rate 1–2, not 0.

## 7. If the deadline lands before any of that

What is defensible to submit from the current run, as-is:

- The `original` row, which reproduces the paper closely.
- Unlearn / Retain / MMLU for CRISP, reported as **a failure to reproduce**: the paper's
  55.26 → 29.67 versus our 55.42 → 55.42, under hyperparameters verified identical to
  Appendix F, with the unspecified step count named as the most likely cause.

What is not defensible: the Overall column (4.66, an artifact of Concept ≈ 0) and the
Fluency/Concept columns generally.
