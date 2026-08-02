# SETUP — what this project is, what we built, and how to run it

Start here if you're new to the repo (or returning to it after a break).

- **This file** — the project, the module map, and how to get running.
- `IMPLEMENTATION_LOG.md` — the narrative: what was built in what order, and the
  judgement calls that aren't in the paper.
- `EXPERIMENTATION_SETUP.md` — the reproduction *plan*: what to run, what it costs,
  and the target numbers.
- `README.md` — the reference: paper-equation → code-symbol table, CLI surface.

---

## 1. What the project is

A from-scratch reimplementation of **CRISP: Persistent Concept Unlearning via
Sparse Autoencoders** (Ashuach, Arad, Mueller, Tutek & Belinkov, ACL 2026 —
`assets/2026.acl-long.82.pdf`), built to reproduce the paper's central claim.

**The method in one paragraph.** Run a *target* corpus (the knowledge to erase —
e.g. WMDP bio-weapons text) and a *retain* corpus (benign, same broad field)
through a language model, reading the residual stream at a few layers through a
pretrained **sparse autoencoder**. Features that fire heavily on the target corpus
and barely on the retain corpus are the concept's features. CRISP then LoRA-
fine-tunes the model so those features' activations go to **zero** on the target
corpus, while two regularisers hold everything else in place: a retention term
pinning hidden states on the retain corpus, and a coherence term pinning them on
20 short benign sentences. Because the edit lives in the *weights*, it persists —
unlike inference-time SAE steering, which is trivially removed.

**Why it's worth reproducing.** The claim is that you can excise a specific
capability while leaving neighbouring, benign competence intact. The evaluation is
built so you can't cheat: the aggregate score (Eq. 12) is a *harmonic* mean over
forgetting, in-domain retention, general MMLU, fluency and concept-avoidance, so a
model that forgets by becoming stupid scores zero.

**What we're targeting.** Not the paper as published (2,400 fine-tuning runs behind
a 200-config Bayesian sweep — hundreds of GPU-hours). The scoped goal is the
**Gemma-2-2B rows of Table 1 and Table 3** at the fixed Appendix F hyperparameters:
a handful of cheap runs that make or break the central CRISP-vs-RMU/ELM comparison.
Llama-3.1-8B is a phase-2 stretch goal needing a rented GPU.

---

## 2. What we have done

Written from the paper, not ported — the authors' release is partial (feature
selection + LoRA optimisation + eval, plus a Harry Potter demo notebook), and WMDP
is not a turnkey script there.

**Built:**

- The full method — Eq. 1 (SAE), Eq. 3–8 (contrastive feature selection), Eq. 9–11
  (the three training losses), Eq. 12 (aggregate score) — each unit-tested against
  hand-computed values.
- SAE loaders for **Gemma Scope**, **Llama Scope**, `sae_lens`, and a random SAE for
  offline smoke tests.
- The complete data pipeline: WMDP/MMLU download, §4.1 corpus cleaning,
  deterministic val/test halving, Appendix D coherence sets, Appendix E prefixes.
- The evaluation harness: zero-shot MCQ scoring, greedy generation, and the LLM
  fluency/concept rater using the paper's verbatim prompts.
- Both **baselines** from Table 1 — RMU (ported directly from Li et al. 2024) and
  ELM (reimplemented from prose).
- A **hyperparameter sweep** over the Appendix F space with the paper's
  geometric-mean selection criterion.
- An **MLX inference backend** for fast local evaluation on Apple silicon.
- 32 tests that run in ~3s with no gated downloads, plus an end-to-end integration
  test on a tiny random Llama.
- Configs carrying the paper's best-found hyperparameters for all four
  (model, domain) pairs.

**Verified locally:** real Gemma Scope SAE loading (layer 14, `d_sae=16384`,
JumpReLU thresholds ≈3.8); WMDP-Cyber corpora (995/4303 docs after cleaning);
WMDP MCQs (994 test items after halving); MMLU loading; end-to-end training,
checkpointing and evaluation on a tiny model.

**Not done:** the Table 1 numbers themselves. `google/gemma-2-2b` is gated and no
valid `HF_TOKEN` has been available here, so no CRISP-vs-baselines comparison
exists yet. The harness is finished; the experiments are not. See §6.

---

## 3. Module map — what each file is for

### Core method

| Module | What it does | Why it exists / why you'd open it |
| --- | --- | --- |
| `src/crisp/sae.py` | `SparseAutoencoder` (JumpReLU / ReLU / TopK) plus loaders for Gemma Scope, Llama Scope, `sae_lens`, and a random SAE. | The SAE is the lens the whole method looks through. Open this when an SAE fails to load, a checkpoint has an unfamiliar key layout, or you want a different width/L0. Handles the annoying reality that every SAE release names and orients its weights differently. |
| `src/crisp/features.py` | Eq. 3–8. Accumulates per-feature activation counts (φ) and magnitudes (A) over each corpus, then takes top-`k` by Δφ and filters by ratio ρ ≥ τ. | The contrastive core — this is *which features get suppressed*. If unlearning is too weak or too destructive, the answer is usually here (`top_k`, `tau`). Results cache to `outputs/features/`. |
| `src/crisp/losses.py` | Eq. 9 (unlearning), Eq. 10 (retention/coherence distance), Eq. 11 (weighted sum). | 68 lines, the entire objective. The clearest single file in the repo — read it to understand the method. |
| `src/crisp/train.py` | The 200-step loop: three forward passes per step (target/retain/coherence), one backward, checkpointing, loss history. | Where the three losses meet an optimiser. Open it to change the schedule, add logging, or debug a loss that won't move. |
| `src/crisp/model.py` | Model loading, `ResidualCapture` (forward hooks on block outputs), `base_model()` (the frozen reference M₀ via `disable_adapter()`), LoRA attachment. | The plumbing everything else stands on. `ResidualCapture` is used by both selection and training, so hooks stay inside the autograd graph. `base_model()` is why there's no second copy of the weights in memory. |

### Data

| Module | What it does | Why it exists / why you'd open it |
| --- | --- | --- |
| `src/crisp/data.py` | WMDP corpus download (incl. the separately-gated `cais/wmdp-bio-forget-corpus`), §4.1 cleaning, WMDP/MMLU MCQ loading, deterministic val/test halving, coherence sets, generation prefixes. | Every dataset-access headache lives here. Cleaning is not cosmetic — markdown headers, citations and URLs are a large fraction of raw WMDP text and would otherwise dominate the activation statistics. |
| `data/coherence/*.json` | 20 benign factual sentences per domain (Appendix D). | The coherence loss's anchor set. Small enough to read; worth reading. |
| `data/prompts/*.json` | 100 generation prefixes per domain (Appendix E). | Inputs to the fluency/concept evaluation. |
| `data/smoke/*.jsonl` | Tiny fake corpora. | Lets the whole pipeline run offline with zero gated downloads. |

### Evaluation

| Module | What it does | Why it exists / why you'd open it |
| --- | --- | --- |
| `src/crisp/eval_mcq.py` | Zero-shot accuracy: argmax over the logits of the four answer-letter tokens after `"Answer:"`. | No generation, no answer parsing, no format-following failures — the measurement is clean. Also handles the left-padding needed to keep the final position aligned across a batch. |
| `src/crisp/eval_gen.py` | Greedy 50-token continuations, then an LLM rater scoring fluency and concept-presence 0–2 with the paper's verbatim Appendix E prompts. | The only paid, non-deterministic part of the pipeline (`--no-judge` skips it). MCQ accuracy alone can't tell "forgot the concept" from "became incoherent"; this can. |
| `src/crisp/evaluate.py` | The harness: WMDP unlearn accuracy, in-domain MMLU retention, full MMLU, fluency/concept, then Eq. 12. | One call produces a full Table 1 row. |
| `src/crisp/metrics.py` | Eq. 12 (`HM(100−U, R, M, 50F, 50C)`) and the Appendix F geometric-mean sweep criterion. | The harmonic mean is the anti-cheating device: zero on any axis ⇒ zero overall. |

### Comparison and search

| Module | What it does | Why it exists / why you'd open it |
| --- | --- | --- |
| `src/crisp/baselines/rmu.py` | RMU (Li et al. 2024): push forget-activations toward a random steering vector, hold retain-activations fixed, update only `down_proj` in a 3-layer window. | The main baseline in Table 1. Ported directly — should be trustworthy. |
| `src/crisp/baselines/elm.py` | ELM (Gandikota et al. 2024): CFG-style erased-distribution target plus retention and fluency terms. | The second baseline. Reimplemented from the paper's prose — treat its numbers as indicative, not exact. |
| `src/crisp/sweep.py` | Optuna TPE (or random search) over the Appendix F space, scored on the **validation** half. | Not needed for the main reproduction (use the fixed hyperparameters). It's the tool for ablations. |

### Infrastructure

| Module | What it does | Why it exists / why you'd open it |
| --- | --- | --- |
| `src/crisp/config.py` | One dataclass per section, YAML loading, `section.field=value` CLI overrides. | Unknown keys **raise** rather than being silently ignored — a config typo becomes a failed run, not a quietly wrong number. Every default is traceable to Appendix F. |
| `src/crisp/cli.py` | `python -m crisp <select\|train\|eval\|baseline\|sweep>`. | The entry point; also where the shared corpus/feature-cache helpers live. |
| `src/crisp/mlx_backend.py` | Evaluation-only `mlx-lm` backend for Apple silicon; duck-types the slice of the HF API that `eval` uses. | Much faster than torch/MPS locally, and 4-bit Gemma-2-2B fits in ~1.5 GB. Inference only — CRISP needs gradients through per-layer activations, which `mlx-lm` doesn't expose. Its docstring explains the right-padding and split-LM-head decisions. |
| `src/crisp/utils.py` | Devices, dtypes, seeding, batching, logging, HF token. | Note the deliberate choice: CUDA → bf16, MPS/CPU → fp32 (bf16 optimiser math is unreliable on MPS). |
| `configs/*.yaml` | The paper's best hyperparameters per (model, domain), plus `smoke.yaml`. | Never edit these to experiment — use `-o` overrides so the paper's settings stay pristine. |
| `scripts/reproduce.sh` | Original model → CRISP → RMU → ELM, all evaluated. | One command produces a complete Table 1 comparison for one config. |
| `tests/` | 32 tests: equations against hand-computed values, config parsing, dataset routing, plus an end-to-end integration run on a tiny random Llama. | ~3s, no gated downloads. If these pass, any failure is data/model access, not code. |

---

## 4. Environment setup

```bash
uv sync --extra dev            # core + pytest
uv sync --extra judge          # + anthropic, for the fluency/concept rater
uv sync --extra sweep          # + optuna
uv sync --extra mlx            # + mlx-lm, Apple-silicon eval backend
```

Python **3.13** (`pyproject.toml` requires `>=3.13`; tests pass on 3.13.7).

> Note: `EXPERIMENTATION_SETUP.md` advises dropping to Python 3.11. That was
> written assuming we'd vendor the authors' `sae_lens`-pinned stack. We don't —
> `sae_lens` is an optional alternative SAE backend, not a dependency — so 3.13
> stands.

### Access you need to provide

| Asset | Status | How |
| --- | --- | --- |
| `cais/wmdp` MCQs, `cais/mmlu` | public | nothing to do |
| `cais/wmdp-corpora` (cyber forget/retain, bio retain) | public | nothing to do |
| `cais/wmdp-bio-forget-corpus` (bio forget) | **gated** | request at the [dataset page](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus), then `export HF_TOKEN=…` — or bypass with `-o data.target_corpus=path/to/bio-forget.jsonl` |
| `google/gemma-2-2b`, `meta-llama/Llama-3.1-8B` | **gated** | accept the licence on the model page, then `export HF_TOKEN=…` |
| Gemma Scope SAEs | public | nothing to do |
| Fluency / Concept scores | needs an LLM rater | `export ANTHROPIC_API_KEY=…`, or run `--no-judge` |

If `hf auth whoami` reports an invalid token, run `hf auth login --force`.

---

## 5. Running it

```bash
# Sanity check — no gated assets, no GPU, seconds
python -m pytest tests/ -q
python -m crisp train -c configs/smoke.yaml --no-eval

# The real thing, one config: original model + CRISP + RMU + ELM, all evaluated
scripts/reproduce.sh configs/gemma2-2b_cyber.yaml

# Or the stages individually
python -m crisp select -c configs/gemma2-2b_bio.yaml      # print salient features
python -m crisp train  -c configs/gemma2-2b_cyber.yaml    # select → train → eval
python -m crisp eval   -c configs/gemma2-2b_cyber.yaml    # untouched model
python -m crisp eval   -c configs/gemma2-2b_cyber.yaml --adapter outputs/runs/gemma2-2b_cyber/adapter

# Fast local eval on Apple silicon
python -m crisp eval -c configs/gemma2-2b_cyber_mlx.yaml --no-judge
```

Override any config field inline rather than editing the YAML:

```bash
python -m crisp train -c configs/gemma2-2b_bio.yaml \
  -o selection.top_k=50 -o train.lambda_scale=20 -o train.lr=3e-5
```

Useful flags: `--no-judge` (skip the paid rater), `--skip-generation` (MCQ metrics
only), `--refresh-features` (ignore the cached feature selection).

**Reading the output.** `unlearn_acc` should *fall* toward 25% (chance);
`retain_acc` and `mmlu` should barely move; `fluency` should stay near the original
model's. `overall` is Eq. 12 over all five.

---

## 6. Current status and next steps

The harness is complete and tested. The experiments are not run — `google/gemma-2-2b`
is gated and no valid `HF_TOKEN` has been available in this environment.

1. `hf auth login --force` with a valid token; accept the `google/gemma-2-2b` licence.
2. Run the **cyber** pair first — it needs no gated dataset:
   `scripts/reproduce.sh configs/gemma2-2b_cyber.yaml`.
3. Request `cais/wmdp-bio-forget-corpus` in parallel. When access lands, the bio
   configs already point at it (`data.target_corpus_repo`), so the same command with
   `configs/gemma2-2b_bio.yaml` just works.
4. Compare against Table 1 — `EXPERIMENTATION_SETUP.md` has the target numbers and
   pass/fail criteria.

Two caveats when you compare: the LLM judge is not the paper's pinned Sonnet 4
(no longer callable), so calibrate fluency/concept against the *original-model*
baseline rather than the paper's absolute numbers; and the val/test MCQ split is
ours, so expect 1–2 points of drift. Neither is a reproduction failure.
`IMPLEMENTATION_LOG.md` §3 lists the rest of the judgement calls.
