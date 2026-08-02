# Implementation Log — what exists in this repo and why

A narrative companion to `README.md`. The README is the reference ("where is Eq. 9?");
this is the walkthrough ("what did we build, in what order, and what were the
non-obvious calls?"). Read top to bottom once and the project should stop feeling
opaque.

Paper: **CRISP: Persistent Concept Unlearning via Sparse Autoencoders**
(Ashuach, Arad, Mueller, Tutek & Belinkov, ACL 2026) — `assets/2026.acl-long.82.pdf`.

---

## 0. The one-paragraph version of the paper

Take a language model. Run a **target corpus** (the thing to forget — e.g. WMDP
bio-weapons text) and a **retain corpus** (benign text from the same broad field)
through it, and read the residual stream at a handful of layers through a
pretrained **sparse autoencoder** (SAE). Some SAE features fire a lot on the
target corpus and barely on the retain corpus — those are the concept's features.
CRISP then **LoRA-fine-tunes the model to push those features' activations to
zero on the target corpus**, while two regularisers hold everything else still:
a retention term pinning hidden states on the retain corpus, and a coherence term
pinning them on 20 short benign sentences. The edit lands in the *weights*, so it
survives — unlike inference-time SAE steering, which is trivially removed.

Three losses (Eq. 9, 10, 11), one feature-selection rule (Eq. 3–8), one
aggregate score (Eq. 12). That's the whole method. Everything below is
engineering around those.

---

## 1. Timeline — what happened, commit by commit

### `0ed56f7` — the paper and ACL style files

Vendored the paper PDF and the ACL LaTeX template. Nothing functional; this is
the reference material the rest of the repo is checked against.

### `e6fa75a` — "feat: harness" (the bulk of the work)

The entire reimplementation landed in one commit: ~2,600 lines across
`src/crisp/`, plus configs, data assets, and tests. Written **from the paper**,
not ported from the authors' repo — their release is partial (feature selection +
LoRA optimisation + eval, plus a Harry Potter demo notebook), and WMDP is not a
turnkey script there.

What that commit established, in dependency order:

| Module | Responsibility |
| --- | --- |
| `config.py` | One `@dataclass` per section (`model`/`sae`/`data`/`selection`/`train`/`eval`), YAML loading, and `section.field=value` CLI overrides. Unknown keys raise instead of being silently ignored — a typo in a config is a failed run, not a wrong number. |
| `data.py` | Corpus download + cleaning, WMDP/MMLU MCQ loading, deterministic val/test halving, coherence sets, generation prefixes. |
| `sae.py` | `SparseAutoencoder` (JumpReLU / ReLU / TopK) + loaders for Gemma Scope, Llama Scope, `sae_lens`, and a random SAE for smoke tests. |
| `model.py` | Model loading, `ResidualCapture` (forward hooks on block outputs), `base_model()` (the frozen reference `M₀`), LoRA attachment. |
| `features.py` | Eq. 3–8: per-feature activation statistics over each corpus, then top-`k` by Δφ filtered by ρ ≥ τ. |
| `losses.py` | Eq. 9 (unlearning), Eq. 10 (retention/coherence distance), Eq. 11 (weighted sum). |
| `train.py` | The 200-step loop, checkpointing, history. |
| `eval_mcq.py` / `eval_gen.py` / `evaluate.py` / `metrics.py` | WMDP + MMLU accuracy, greedy generation + LLM-rater fluency/concept, the full harness, Eq. 12. |
| `baselines/rmu.py`, `baselines/elm.py` | The two comparison methods from Table 1. |
| `sweep.py` | Appendix F hyperparameter search (Optuna TPE if installed, random search otherwise). |
| `cli.py` | `python -m crisp <select\|train\|eval\|baseline\|sweep>`. |

Plus `configs/` with the paper's best-found hyperparameters per (model, domain),
`data/coherence/` (Appendix D, 20 sentences/domain), `data/prompts/` (Appendix E,
100 prefixes/domain), and `data/smoke/` (tiny corpora so the pipeline can be
exercised offline with no gated downloads).

### `309f0ba` — MLX inference backend

Added an **evaluation-only** second backend so MCQ scoring and generation run on
Apple-silicon GPUs via `mlx-lm` instead of torch/MPS (much faster; 4-bit Gemma-2-2B
is ~1.5 GB). `select`/`train`/`baseline`/`sweep` explicitly refuse to run under
`backend: mlx` — CRISP differentiates through per-layer residual activations, and
`mlx-lm` exposes neither forward hooks nor that autograd surface.

Two things this forced (both documented at the top of `mlx_backend.py`):

- MCQ batches are **right**-padded here, versus left-padded on the torch path.
  `mlx-lm` builds its own causal mask and accepts no attention mask, so a padding
  mask can't be supplied. Under causal attention a *trailing* pad cannot reach an
  earlier real token, so right padding is exact; left padding would leak pad
  positions into every query.
- The backbone and the LM head are invoked separately, because Gemma's 256k-row
  logit matrix over a padded `[B, L]` batch is several GB and only the final real
  position is ever used.

The dispatch is duck-typed: `eval_mcq` checks for `final_logits`, `eval_gen`
checks for `generate_texts`. Nothing else in the pipeline knows a backend exists.

### `51468ac` — `EXPERIMENTATION_SETUP.md`

The reproduction *plan*, as distinct from the code: don't reproduce the paper as
published (2,400 fine-tuning runs behind a 200-config Bayesian sweep — hundreds of
GPU-hours). Reproduce the **Gemma-2-2B rows of Table 1 and Table 3** at the fixed
Appendix F hyperparameters. That's a handful of sub-$2 runs and it makes or breaks
the paper's central comparison.

### 2026-08-02 (uncommitted) — the WMDP bio-forget corpus

Fixed a wrong assumption in the original `data.py`: it expected all four WMDP
corpora to be configs of `cais/wmdp-corpora`. They aren't. The bio *forget*
corpus is not published there at all — it lives in its own gated repo,
**`cais/wmdp-bio-forget-corpus`**, as a single default parquet config.

Changes:

- `data.py`: `WMDP_CORPUS_CONFIGS` → `WMDP_CORPUS_SOURCES`, a `(repo_id, config_name)`
  map. `bio_target` → `("cais/wmdp-bio-forget-corpus", None)`; the other three stay
  on `cais/wmdp-corpora` configs. `_download_wmdp_corpus` takes a `repo_override`
  and reports the right repo URL when access fails.
- `config.py`: new `data.target_corpus_repo` / `data.retain_corpus_repo` (HF repo
  override — distinct from the pre-existing `target_corpus`/`retain_corpus`, which
  are *local file paths*).
- `cli.py`, `sweep.py`: pass them through, so `select`/`train`/`baseline`/`sweep`
  all use it.
- The three bio configs pin `target_corpus_repo` explicitly.

Verified: cyber-forget and bio-retain load and clean correctly through the new
code path. Bio-forget itself returns `GatedRepoError` — access has to be requested
on the dataset page and a valid `HF_TOKEN` exported. (The token currently stored
on this machine is invalid; `hf auth login --force` fixes that.)

---

## 2. What actually happens when you run `python -m crisp train`

Follow this once and the module layout explains itself.

### Step 1 — config (`cli.py::_load_config`)

YAML → nested dataclasses → apply any `-o section.field=value` overrides. All
downstream code reads `cfg`, never the environment.

### Step 2 — model + SAEs (`model.py::load_model_and_tokenizer`, `sae.py::load_saes`)

Device/dtype resolution is deliberate: **CUDA → bf16, MPS/CPU → fp32**, because
bf16 optimiser math is unreliable on MPS.

For each of the 6 suppressed layers (`model.sae_layers`, `[4,6,8,10,12,14]` for
Gemma-2-2B), one pretrained SAE is downloaded and frozen. `sae.filename_template:
auto` lists the Gemma Scope repo and picks the release whose average L0 is nearest
`l0_target: 100` — because the *canonical* repo is access-controlled, and the
canonical rule is "L0 ≈ 100" anyway.

### Step 3 — corpora (`cli.py::_load_corpora` → `data.py::load_corpus`)

Download (or read a local `.jsonl`), then `clean_document`: strip markdown headers,
image/inline links, URLs and `[12]`/`(Smith, 2020)`-style citations, drop non-ASCII,
truncate to 1,000 chars (§4.1). Documents under 50 chars are dropped. Bio samples
5,000 docs at a fixed seed; cyber uses all ~986.

Also loaded: the 20-sentence coherence set for the domain.

### Step 4 — feature selection (`features.py::run_selection`)

The contrastive core, Eq. 3–8. For up to `selection.max_docs` (500) documents per
corpus:

1. Forward pass with `ResidualCapture` hooks on each suppressed block's **output**
   (`hook_resid_post` — what Gemma/Llama Scope SAEs are trained on).
2. Encode every real token's hidden state through the layer's SAE.
3. Accumulate two `[d_sae]` vectors: `count` = tokens where the feature fired
   (φ, Eq. 3), `total` = summed activation magnitude (A, Eq. 5).

Then per layer: `Δφ = count_target − count_retain·scale` (Eq. 4), take the top
`k` features (Eq. 7), and keep only those whose activation ratio
`ρ = A_target / (A_retain·scale + ε)` clears `τ = 3` (Eq. 6, 8).

Result is cached to `outputs/features/<model>__<domain>__k<k>_tau<τ>.json`, so
re-running `train` skips this stage unless `--refresh-features`.

### Step 5 — LoRA (`model.py::attach_lora`)

Adapters go on all 7 projection matrices of blocks `[3–9]` — the paper's "early
optimisation layers". Note these are a *different* set from the SAE-suppressed
layers: you read at `[4,…,14]` and write at `[3,…,9]`.

### Step 6 — the training loop (`train.py::train_crisp`)

200 steps. Each step does **three forward passes** and one backward:

| Term | Corpus | What it computes |
| --- | --- | --- |
| `l_unlearn` (Eq. 9) | target | Mean over tokens of `mean(salient feature acts) + λ·mean(all feature acts)`, averaged over the 6 SAE layers. The λ term (λ=30) is a whole-dictionary penalty that stops the model from simply routing the concept into unselected features. |
| `l_retain` (Eq. 10) | retain | Mean over tokens of `‖h_M − h_M₀‖²`, averaged over the same 6 layers. |
| `l_coherence` | 20 curated sentences | Same distance, but at the **final** layer only. |

Combined as `α·unlearn + β·retain + γ·coherence` with β=0.99, γ=0.01, α=1−β=0.01
(Eq. 11). The weighting looks lopsided but isn't — the retention term is a squared
L2 norm over `d_model`, so it's numerically large; the unlearning term is a mean
activation.

**The reference `M₀` is not a second copy of the model.** `base_model()` is a
context manager that calls PEFT's `disable_adapter()`, so `h_M₀` comes from the
same weights with the LoRA path switched off. `tests/test_integration.py` asserts
the adapter-disabled logits equal the pre-LoRA logits exactly.

Output: `outputs/runs/<run_name>/` with `adapter/`, `features.json`, `config.json`,
`history.json`.

### Step 7 — evaluation (`evaluate.py::evaluate_model`)

Four metrics, on the held-out **test** half of each MCQ set:

- **`unlearn_acc`** — WMDP accuracy in the domain. *Lower is better.* 25% = chance.
- **`retain_acc`** — MMLU on the two in-domain subjects (e.g. high-school +
  college biology). Higher is better.
- **`mmlu`** — full MMLU, general capability.
- **`fluency` / `concept`** — an LLM rater scores 100 greedy 50-token
  continuations 0–2 each, using the paper's verbatim Appendix E prompts.

MCQ scoring is zero-shot argmax over the logits of the four answer-letter tokens
after `"Answer:"` — no generation, no parsing. Left padding keeps the final
position aligned across a batch.

Aggregated by `metrics.py::overall_score` = Eq. 12 = `HM(100−U, R, M, 50F, 50C)`.
A **harmonic** mean, deliberately: a method that scores 0 on any single axis
scores 0 overall, so you can't win by lobotomising the model.

---

## 3. Decisions that aren't in the paper

These are the places where the paper underspecifies and a call had to be made.
If a number looks off later, suspect these first.

1. **Δφ is normalised by corpus size.** Eq. 4 subtracts raw activation counts, but
   the corpora differ in token count (cyber-retain is ~4× cyber-forget). Counts are
   rescaled by the token-count ratio before subtracting, so Δφ and ρ measure
   per-token *rates*. Without this the selection would largely rank corpus size.
   (`features.py::select_features`, tested.)

2. **`train.retain_reduction` defaults to `sqnorm`** — Eq. 10 verbatim, mean over
   tokens of the squared L2 norm. Setting it to `mse` divides by `d_model`, which
   rescales the retention term without changing its direction. Kept as a knob
   because it interacts with the β=0.99 weighting.

3. **Hook site is the un-normalised residual stream.** HF applies the final RMSNorm
   to the *last* entry of `output_hidden_states`; the hook reads block outputs
   directly to avoid that.

4. **The LLM judge is not the paper's.** The paper pins Claude Sonnet 4
   `2025-05-14`, which is no longer callable. `eval.judge_model` defaults to a
   current model. **Fluency/Concept numbers are therefore not directly comparable
   to the paper's** unless you also re-score the original-model baseline with the
   same judge and compare deltas. Do that.

5. **The val/test MCQ split is ours, not theirs.** `split_half` shuffles at a fixed
   seed and halves. Expect 1–2 points of drift on unlearn/retain accuracy. Not a
   reproduction failure.

6. **ELM is a reimplementation from prose**, not a port — CFG-style erasure target
   plus retention and fluency terms. Treat its numbers as indicative. RMU follows
   Li et al. (2024) directly and should be trustworthy.

7. **The sweep is smaller than the paper's.** They run Bayesian optimisation over
   200 configs per method (2,400 runs total). `sweep.py` implements the same search
   space and the same geometric-mean selection criterion, but you're expected to
   run the fixed Appendix F hyperparameters instead. The sweep exists for ablations.

---

## 4. Where the project actually stands

**Working and verified locally:**

- All 32 unit tests pass in ~3s with no gated downloads. They check the equations
  numerically against hand-computed values — Eq. 9 term by term, Eq. 10 reductions,
  Eq. 11 weighting, Eq. 12 harmonic mean, top-k/τ selection including the
  corpus-size normalisation, corpus cleaning, val/test splitting.
- `tests/test_integration.py` runs the *whole* pipeline on a tiny random Llama —
  capture, LoRA, selection, training, MCQ scoring, generation — and asserts the
  unlearning loss actually decreases.
- Real Gemma Scope SAE loading (layer 14, `d_sae=16384`, JumpReLU thresholds ≈3.8).
- WMDP-Cyber forget/retain corpora (995/4303 docs after cleaning), WMDP MCQs
  (994 test items after halving), MMLU subject loading.

**Not done yet — this is the honest gap:**

- **No Table 1 numbers exist.** `google/gemma-2-2b` is gated and no valid `HF_TOKEN`
  has been available in this environment, so no CRISP-vs-RMU/ELM comparison has
  been produced. The harness is finished; the experiments are not.
- The bio half is additionally blocked on `cais/wmdp-bio-forget-corpus` access.
- Llama-3.1-8B is a phase-2 stretch goal — it needs a rented GPU.

**Unblocking order:**

1. `hf auth login --force` with a valid token (the stored one is invalid), and
   accept the `google/gemma-2-2b` licence.
2. Run the **cyber** pair end-to-end — it needs no gated dataset:
   `scripts/reproduce.sh configs/gemma2-2b_cyber.yaml`
   (original model → CRISP → RMU → ELM, all four evaluated on the test split).
3. Request `cais/wmdp-bio-forget-corpus` in parallel; when it lands, the bio
   configs already point at it, so the same command with
   `configs/gemma2-2b_bio.yaml` just works.
4. Compare against Table 1. `EXPERIMENTATION_SETUP.md` has the target numbers and
   the pass/fail criteria.

---

## 5. Fast path for a sanity check

No gated assets, no GPU, ~seconds:

```bash
python -m pytest tests/ -q                      # equations + tiny end-to-end run
python -m crisp train -c configs/smoke.yaml     # random SAE, local tiny corpora
```

If those pass, the machinery is intact and any problem is data/model access, not code.
