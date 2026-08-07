# crisp-reproducibility

Reproduction of **CRISP: Persistent Concept Unlearning via Sparse Autoencoders**
(Ashuach, Arad, Mueller, Tutek & Belinkov, ACL 2026) —
[paper](https://aclanthology.org/2026.acl-long.82.pdf).

CRISP identifies SAE features that fire on a *target* corpus but not on a *retain*
corpus, then LoRA-fine-tunes the model to suppress those features on the target
corpus while pinning its hidden states on benign text. Unlike inference-time SAE
steering, the edit lives in the weights.

## How this is run

The laptop here is an M4 with no CUDA, so the work is split:

| Where | What |
| --- | --- |
| Your machine | `python -m crisp fetch` — materialise the corpora and benchmarks into `data/` — plus the tests and the smoke config |
| Google Colab | feature selection, CRISP training, the RMU/ELM baselines, evaluation, figures — [`notebooks/crisp_colab.ipynb`](notebooks/crisp_colab.ipynb) |
| This repo | the results table and figures, committed back from the notebook with a GitHub token |

## Documentation

| File | What it's for |
| --- | --- |
| [`SETUP.md`](SETUP.md) | **Start here.** Project description, what each module is for, environment setup, how to run. |
| [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) | What was built in what order, what happens on `crisp train` step by step, and the judgement calls that aren't in the paper. |
| [`EXPERIMENTATION_SETUP.md`](EXPERIMENTATION_SETUP.md) | The reproduction plan: what to run, what it costs, target numbers. |

## Install

```bash
uv sync --extra dev            # add --extra sweep for Optuna
```

Locally this is for `crisp fetch`, `pytest` and `configs/smoke.yaml`; the GPU
dependencies are installed inside the Colab notebook around Colab's own torch.

### Access you need to provide

| Asset | Status | How |
| --- | --- | --- |
| `cais/wmdp` MCQs, `cais/mmlu` | public | nothing to do |
| `cais/wmdp-corpora` cyber forget/retain, bio retain | public | nothing to do |
| `cais/wmdp-bio-forget-corpus` (bio forget) | gated | request at the [dataset page](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus), then set `HF_TOKEN` (or pass `-o data.target_corpus=path/to/bio-forget.jsonl`) |
| `google/gemma-2-2b`, `meta-llama/Llama-3.1-8B` | gated | accept the licence on the model page, then set `HF_TOKEN` |
| Gemma Scope SAEs | public | nothing to do |
| `Qwen/Qwen3-4B-Thinking-2507` (fluency/concept rater) | public | downloaded on first judged eval (~8 GB); skip with `--no-judge` |
| GitHub personal access token | yours | fine-grained, this repo, **Contents: read and write** — lets the notebook push results back |

Locally, credentials are read from a `.env` file at the repo root
(`echo 'HF_TOKEN=hf_...' >> .env`), so nothing needs exporting per shell. On Colab
they come from the notebook's 🔑 Secrets panel instead.


## Fetch the datasets

This is the one step that runs on your machine. Every corpus and benchmark is
materialised under `data/` first; selection, training and evaluation then read
from there.

```bash
python -m crisp fetch                  # both domains
python -m crisp fetch --domain bio     # or just one
```

This writes `data/wmdp/{bio,cyber}_{target,retain}.jsonl`, `data/mcq/*.jsonl`
and a `data/MANIFEST.json` recording the source repo and row count of each file.
The fetched data is gitignored (the bio forget corpus is gated and not
redistributable), so it travels to Colab through your Drive: upload the whole
`data/` folder to `MyDrive/crisp/data/` and the notebook mounts it from there.
That pins the run to the exact files you fetched rather than to whatever the Hub
serves that day. (`reproduce.sh` will fetch on the Colab machine too if the folder
isn't there — it needs the same gate approvals.)

## Run it on Colab

[`notebooks/crisp_colab.ipynb`](notebooks/crisp_colab.ipynb) is the whole pipeline:
it clones this repo with a GitHub token, mounts the datasets from Drive, reads
`HF_TOKEN` from Colab secrets, installs the dependencies around Colab's
preinstalled torch, runs `scripts/reproduce.sh`, renders the table and the
figures, and **commits `artifacts/results/` and `artifacts/figures/` straight back
to this repo**. `CONFIG`, `STAGES` and `REPO` at the top of the notebook select the
run.

| Secret (🔑 Colab sidebar) | Needed for |
| --- | --- |
| `HF_TOKEN` | the gated `google/gemma-2-2b` weights and the gated bio forget corpus |
| `GITHUB_TOKEN` | cloning, and pushing the results back — fine-grained, this repo only, **Contents: read and write** |

| Runtime | VRAM | What fits |
| --- | --- | --- |
| T4 (free) | ~15 GB | `smoke.yaml`; `gemma2-2b` eval, and training only tightly |
| L4 (Pro) | ~22 GB | full `gemma2-2b_{bio,cyber}` in bf16 — the target to aim for |
| A100 40 GB | 40 GB | as above comfortably; `llama31-8b` is possible but tight |

Rough wall clock on an L4: 3–5 hours for all four stages, of which
`--stages original,crisp` — the headline comparison — is about half. Colab
disconnects well before that, so the notebook symlinks the HF cache and `data/`
onto Drive and copies `artifacts/` back and forth; because each stage is skipped
when its result JSON already exists, re-running the notebook resumes rather than
restarting.

The token never lands on disk: the clone URL carries it, then the remote is reset
to the plain HTTPS URL, and the push re-attaches it for that one command.

## What the pipeline does

`scripts/reproduce.sh` is what the notebook actually runs, and it works the same
from any CUDA box:

```bash
scripts/reproduce.sh configs/gemma2-2b_bio.yaml
```

That is: fetch datasets → evaluate the original model → train CRISP → train RMU →
train ELM → write the comparison table → render the figures. Results land in
`artifacts/results/` (one `<run>__<split>.json` per model, plus a regenerated
`summary.json` and `README.md` table) and `artifacts/figures/`.

It picks the dtype for whatever card it finds — bf16 on Ampere and newer, float32
on a T4 — and subsamples the general-MMLU utility column to 2 questions per
subject. That column is 14k questions and otherwise dominates every evaluation,
while all 57 subjects stay represented; the WMDP and in-domain-MMLU columns, the
ones the paper's claims rest on, stay at full size.

| Flag | Effect |
| --- | --- |
| `--full-mmlu` | keep the general-MMLU column at full size (several extra hours) |
| `--no-fetch` | trust what is already in `data/` (e.g. mounted from Drive) |
| `--fresh` | re-run stages whose results already exist (default is to resume) |
| `--stages original,crisp` | run only some of `original,crisp,rmu,elm` |
| anything else | forwarded to `crisp` (e.g. `--skip-generation`, `-o train.steps=50`) |

The run is resumable: each stage is skipped when its result file is already in
`artifacts/results/`, so an interrupted run picks up where it stopped.

Before committing GPU-hours, check the plumbing in about a minute on a tiny random
model that needs no gated downloads and no GPU — this one is worth running locally:

```bash
scripts/reproduce.sh configs/smoke.yaml
```

## Figures

```bash
python -m crisp plots            # -> artifacts/figures/
```

Rendered from the same result JSONs `crisp report` aggregates, so the figures and
the table can never disagree:

- `metrics_<config>.png` — every Table 1 column, one bar group per method (the 0–2
  rater columns rescaled by 50, as in Eq. 12).
- `tradeoff_<config>.png` — WMDP accuracy against in-domain MMLU, i.e. the shape of
  the paper's actual claim: the forget axis drops without dragging utility with it.
  Bottom-right is where a good method lands; the dashed line is 25% chance.
- `training_<run>.png` — the loss and the three Eq. 11 terms per step, one panel
  each because they live on very different scales.

`artifacts/figures/` and `artifacts/results/{README.md,summary.json}` are the only
artifacts tracked in git — adapters, feature caches and the datasets stay out.

## Individual commands

```bash
# Full pipeline: feature selection -> LoRA training -> test-split evaluation
python -m crisp train -c configs/gemma2-2b_cyber.yaml

# Feature selection only (prints the salient feature ids per layer)
python -m crisp select -c configs/gemma2-2b_bio.yaml

# Evaluate the untouched model, or a saved adapter
python -m crisp eval -c configs/gemma2-2b_cyber.yaml
python -m crisp eval -c configs/gemma2-2b_cyber.yaml --adapter artifacts/runs/gemma2-2b_cyber/adapter

# Baselines
python -m crisp baseline -c configs/gemma2-2b_cyber.yaml --method rmu
python -m crisp baseline -c configs/gemma2-2b_cyber.yaml --method elm

# Hyperparameter search on the validation split (Appendix F space)
python -m crisp sweep -c configs/gemma2-2b_bio.yaml --trials 200

# Rebuild artifacts/results/{summary.json,README.md} from the result JSONs
python -m crisp report
```

Any config field can be overridden inline:

```bash
python -m crisp train -c configs/gemma2-2b_bio.yaml \
  -o selection.top_k=50 -o train.lambda_scale=20 -o train.lr=3e-5
```

```bash
# Rebuild artifacts/figures from the same result JSONs
python -m crisp plots
```

Add `--no-judge` to skip the local LLM rater, or `--skip-generation` to skip
generation entirely and report MCQ metrics only.

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
- **Hardware.** `reproduce.sh` resolves dtype from the card: bf16 on Ampere and
  newer, float32 on pre-Ampere (a T4) and on CPU. Training runs without a gradient
  scaler, so float16 there would give silent NaNs where float32 gives a clean OOM.
  Gemma-2-2B in bf16 is comfortable on an L4; Llama-3.1-8B wants an A100 — the
  paper used RTX 6000 Ada 49 GB cards.

## Tests

```bash
python -m pytest tests/ -q          # 43 tests, ~2s, no gated downloads
```

`tests/test_crisp.py` checks the equations numerically against hand-computed
values (Eq. 9 term by term, Eq. 10 reductions, Eq. 11 weighting, Eq. 12
harmonic mean, top-`k`/`τ` selection including the corpus-size normalisation,
corpus cleaning, val/test splitting). `tests/test_integration.py` runs the whole
pipeline — capture, LoRA, selection, training, MCQ scoring, generation — on a
tiny random Llama, and asserts the unlearning loss actually decreases.

## Verification status

Verified locally: SAE loading against real Gemma Scope weights (layer 14,
`d_sae=16384`, JumpReLU thresholds ≈3.8); dataset fetch for both domains
(bio 24453/60887, cyber 1000/4473 raw forget/retain docs; WMDP MCQs 1273/1987;
MMLU 14042) into `data/`; `scripts/reproduce.sh` all the way through
original → CRISP → RMU → ELM → results table → figures.

Not run here: the Table 1 numbers themselves. The full-scale run has only been
exercised on the tiny smoke model, so no CRISP-vs-RMU/ELM comparison on
`gemma-2-2b` exists yet. Fetch the data locally, then open
`notebooks/crisp_colab.ipynb` on an L4 and run it — it will populate
`artifacts/results/` and `artifacts/figures/` and push them back here.

## Layout

```
configs/            paper hyperparameters per (model, domain) + a smoke config
data/coherence/     Appendix D coherence sets
data/prompts/       Appendix E generation prefixes
data/smoke/         tiny corpora for the offline pipeline check
data/wmdp/          forget/retain corpora written by `crisp fetch` (gitignored)
data/mcq/           WMDP + MMLU benchmarks written by `crisp fetch` (gitignored)
data/MANIFEST.json  source repo and row count of every fetched file
artifacts/results/  one JSON per evaluated model + summary.json + README.md table
artifacts/figures/  metric bars, forget/retain trade-off, training curves (tracked)
artifacts/runs/     adapters, training histories, selected features
notebooks/          crisp_colab.ipynb -- the GPU runner, results pushed back here
scripts/            reproduce.sh -- the one-command pipeline
src/crisp/
  config.py         dataclass config + YAML/CLI overrides
  data.py           corpora, cleaning, WMDP/MMLU MCQs, coherence sets
  fetch.py          materialises every dataset under data/
  report.py         aggregates artifacts/results into the Table 1 comparison
  plots.py          figures for those same results
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
  cli.py            python -m crisp <fetch|select|train|eval|baseline|sweep|report|plots>
```


## References:

1. https://arxiv.org/pdf/2410.19278#page=12.18
2. https://arxiv.org/abs/2508.13650
