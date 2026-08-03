# crisp-reproducibility

Reproduction of **CRISP: Persistent Concept Unlearning via Sparse Autoencoders**
(Ashuach, Arad, Mueller, Tutek & Belinkov, ACL 2026) —
[paper](https://aclanthology.org/2026.acl-long.82.pdf).

CRISP identifies SAE features that fire on a *target* corpus but not on a *retain*
corpus, then LoRA-fine-tunes the model to suppress those features on the target
corpus while pinning its hidden states on benign text. Unlike inference-time SAE
steering, the edit lives in the weights.

## Documentation

| File | What it's for |
| --- | --- |
| [`SETUP.md`](SETUP.md) | **Start here.** Project description, what each module is for, environment setup, how to run. |
| [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) | What was built in what order, what happens on `crisp train` step by step, and the judgement calls that aren't in the paper. |
| [`EXPERIMENTATION_SETUP.md`](EXPERIMENTATION_SETUP.md) | The reproduction plan: what to run, what it costs, target numbers. |
| `README.md` (this file) | Reference: paper-equation → code-symbol table, CLI surface, implementation notes. |

## Install

```bash
uv sync --extra dev            # add --extra judge for the LLM rater, --extra sweep for Optuna
```

### Access you need to provide

| Asset | Status | How |
| --- | --- | --- |
| `cais/wmdp` MCQs, `cais/mmlu` | public | nothing to do |
| `cais/wmdp-corpora` cyber forget/retain, bio retain | public | nothing to do |
| `cais/wmdp-bio-forget-corpus` (bio forget) | gated | request at the [dataset page](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus), then `export HF_TOKEN=…` (or pass `-o data.target_corpus=path/to/bio-forget.jsonl`) |
| `google/gemma-2-2b`, `meta-llama/Llama-3.1-8B` | gated | accept the licence on the model page, then `export HF_TOKEN=…` |
| Gemma Scope SAEs | public | nothing to do |
| Fluency / Concept scores | needs an LLM rater | `export ANTHROPIC_API_KEY=…` (or run with `--no-judge`) |

## Run

```bash
# Full pipeline: feature selection -> LoRA training -> test-split evaluation
python -m crisp train -c configs/gemma2-2b_cyber.yaml

# Feature selection only (prints the salient feature ids per layer)
python -m crisp select -c configs/gemma2-2b_bio.yaml

# Evaluate the untouched model, or a saved adapter
python -m crisp eval -c configs/gemma2-2b_cyber.yaml
python -m crisp eval -c configs/gemma2-2b_cyber.yaml --adapter outputs/runs/gemma2-2b_cyber/adapter

# Baselines
python -m crisp baseline -c configs/gemma2-2b_cyber.yaml --method rmu
python -m crisp baseline -c configs/gemma2-2b_cyber.yaml --method elm

# Hyperparameter search on the validation split (Appendix F space)
python -m crisp sweep -c configs/gemma2-2b_bio.yaml --trials 200

# Everything for one config (original model + CRISP + both baselines)
scripts/reproduce.sh configs/gemma2-2b_cyber.yaml
```

Any config field can be overridden inline:

```bash
python -m crisp train -c configs/gemma2-2b_bio.yaml \
  -o selection.top_k=50 -o train.lambda_scale=20 -o train.lr=3e-5
```

Add `--no-judge` to skip the paid LLM rater, or `--skip-generation` to skip
generation entirely and report MCQ metrics only.

## Local inference on Apple silicon (mlx-lm)

`crisp eval` has a second backend that runs the model through
[mlx-lm](https://github.com/ml-explore/mlx-lm) instead of torch/MPS — much faster
on an M-series Mac, and 4-bit weights keep Gemma-2-2B at ~1.5 GB.

```bash
uv sync --extra mlx

python -m crisp eval -c configs/gemma2-2b_cyber_mlx.yaml --no-judge
python -m crisp eval -c configs/gemma2-2b_bio_mlx.yaml   --no-judge

# or switch any existing config over at the command line
python -m crisp eval -c configs/gemma2-2b_cyber.yaml --backend mlx --no-judge
```

The backend is selected by `model.backend: torch|mlx`, with `model.mlx_name`
naming the MLX checkpoint (defaults to the `mlx-community` mirror of
`model.name`).

**It is inference-only.** `select`, `train`, `baseline` and `sweep` exit with an
error under `backend: mlx`: CRISP differentiates through per-layer residual
activations, and mlx-lm exposes neither forward hooks nor the autograd surface
that needs. Two consequences:

* `--adapter` accepts mlx-lm adapters only; a PEFT adapter produced by
  `crisp train` is rejected with a pointer back to the torch backend.
* Quantised weights perturb the residual stream, so 4-bit MCQ accuracies drift
  from the bf16 numbers in the paper. Use it for fast iteration and sanity
  checks; report from the torch path.

Implementation notes live at the top of `src/crisp/mlx_backend.py` — chiefly why
MCQ batches are right-padded here (mlx-lm builds its own causal mask and accepts
no attention mask) and why the LM head is applied only to the final position
(Gemma's 256k-row logit matrix over a padded batch is several GB otherwise).

## How the code maps onto the paper

| Paper | Code |
| --- | --- |
| Eq. 1 — SAE encode/decode (JumpReLU, ReLU, TopK) | `crisp/sae.py::SparseAutoencoder` |
| Eq. 3–4 — activation count `φ`, difference `Δφ` | `crisp/features.py::corpus_statistics` |
| Eq. 5–6 — cumulative activation `A`, ratio `ρ` | `crisp/features.py::select_features` |
| Eq. 7–8 — top-`k` then `ρ ≥ τ` filter | `crisp/features.py::select_features` |
| Eq. 9 — unlearning loss | `crisp/losses.py::unlearning_loss` |
| Eq. 10 — retention loss | `crisp/losses.py::representation_distance` |
| Coherence loss (§3.3, final layer) | `crisp/train.py::train_crisp` |
| Eq. 11 — weighted total | `crisp/losses.py::total_loss` |
| Eq. 12 — Overall = HM(100−U, R, M, 50F, 50C) | `crisp/metrics.py::overall_score` |
| §4.1 — corpus cleaning, 1000-char truncation, val/test halving | `crisp/data.py` |
| App. D — coherence sets (20 sentences/domain) | `data/coherence/*.json` |
| App. E — 100 prefixes/domain, greedy 50-token decoding, rater prompts | `data/prompts/*.json`, `crisp/eval_gen.py` |
| App. F — search space + geometric-mean selection criterion | `crisp/sweep.py`, `crisp/metrics.py::selection_score` |

The configs in `configs/` carry the paper's best-found hyperparameters
(Appendix F): Gemma-2-2B suppresses SAE layers `[4,6,8,10,12,14]`
(Bio `k=30, λ=30, r=8`; Cyber `k=50, λ=20, r=4`), Llama-3.1-8B uses
`[4,6,…,28]` for Bio (`k=10, λ=40, r=8`) and `[4,6,…,18]` for Cyber
(`k=50, λ=30, r=4`); all use lr `4e-5`, `τ=3`, `β=0.99`, `γ=0.01`, `α=1−β`, and
LoRA on blocks `[3–9]`.

## Implementation notes

- **No second copy of the model.** The frozen reference `M₀` in Eq. 10 is
  obtained by disabling the LoRA adapters (`crisp/model.py::base_model`) rather
  than holding a second set of weights. `tests/test_integration.py` asserts the
  adapter-disabled logits equal the pre-LoRA logits exactly.
- **Hook site.** Activations are captured at each block's output
  (`hook_resid_post`), which is what Gemma Scope / Llama Scope SAEs are trained
  on. Note HF applies the final RMSNorm to the *last* entry of
  `output_hidden_states`; the hook deliberately reads the un-normalised stream.
- **Δφ normalisation.** Eq. 4 subtracts raw counts. Because the target and
  retain corpora differ in token count (cyber-retain is ~4× the cyber-forget
  corpus), counts are rescaled by the token-count ratio before subtracting, so
  `Δφ` and `ρ` measure per-token rates. Without this the metric would largely
  rank corpus size.
- **`train.retain_reduction`** defaults to `sqnorm`, i.e. Eq. 10 verbatim (mean
  over tokens of the squared L2 norm). Set it to `mse` to divide by `d_model`,
  which rescales the retention term without changing its direction.
- **SAE selection.** `google/gemma-scope-2b-pt-res-canonical` is access-
  controlled, so the configs point at the public `google/gemma-scope-2b-pt-res`
  and `sae.filename_template: auto` picks the release whose average L0 is
  nearest `sae.l0_target` (100, matching the canonical rule).
- **Llama Scope** publishes one repo per layer with varying key layouts; the
  loader normalises common names and orientations, and `sae.source: sae_lens`
  is available as an alternative backend if you have `sae_lens` installed.
- **Baselines.** RMU follows Li et al. (2024) directly. ELM is reimplemented
  from its paper description (CFG-style erasure target + retention KL + fluency
  term); treat its numbers as indicative rather than an exact reproduction.
- **Hardware.** Defaults are device-agnostic (`cuda` → bf16, MPS/CPU → fp32,
  since bf16 optimiser math is unreliable on MPS). Gemma-2-2B fits on a 24 GB
  Apple-silicon machine; Llama-3.1-8B wants a real GPU — the paper used RTX 6000
  Ada 49 GB cards.

## Tests

```bash
python -m pytest tests/ -q          # 28 tests, ~2s, no gated downloads
```

`tests/test_crisp.py` checks the equations numerically against hand-computed
values (Eq. 9 term by term, Eq. 10 reductions, Eq. 11 weighting, Eq. 12
harmonic mean, top-`k`/`τ` selection including the corpus-size normalisation,
corpus cleaning, val/test splitting). `tests/test_integration.py` runs the whole
pipeline — capture, LoRA, selection, training, MCQ scoring, generation — on a
tiny random Llama, and asserts the unlearning loss actually decreases.

## Verification status

Verified locally: SAE loading against real Gemma Scope weights (layer 14,
`d_sae=16384`, JumpReLU thresholds ≈3.8); WMDP-Cyber forget/retain corpora
(995/4303 docs after cleaning), WMDP MCQs (994 test items after halving) and
MMLU subject loading; end-to-end training, checkpointing and evaluation on a
tiny model.

Not run here: the Table 1 numbers themselves. `google/gemma-2-2b` is gated and
no `HF_TOKEN` was available in this environment, so no CRISP-vs-RMU/ELM
comparison has been produced yet. Set `HF_TOKEN` and run `scripts/reproduce.sh`
to generate them.

## Layout

```
configs/            paper hyperparameters per (model, domain) + a smoke config
data/coherence/     Appendix D coherence sets
data/prompts/       Appendix E generation prefixes
data/smoke/         tiny corpora for the offline pipeline check
scripts/            reproduce.sh
src/crisp/
  config.py         dataclass config + YAML/CLI overrides
  data.py           corpora, cleaning, WMDP/MMLU MCQs, coherence sets
  sae.py            SAE module + Gemma Scope / Llama Scope / sae_lens loaders
  model.py          model loading, residual capture, LoRA, frozen-reference ctx
  features.py       Eq. 3-8 contrastive feature selection
  losses.py         Eq. 9-11
  train.py          training loop and checkpointing
  eval_mcq.py       WMDP / MMLU accuracy
  eval_gen.py       generation + fluency/concept rater
  evaluate.py       full evaluation harness
  metrics.py        Eq. 12 and the Appendix F selection criterion
  sweep.py          hyperparameter search
  baselines/        RMU, ELM
  cli.py            python -m crisp <select|train|eval|baseline|sweep>
```


## References:

1. https://arxiv.org/pdf/2410.19278#page=12.18
2. https://arxiv.org/abs/2508.13650
