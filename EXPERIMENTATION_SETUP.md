# Experimentation Setup — Reproducing CRISP

Reproduction plan for **CRISP: Persistent Concept Unlearning via Sparse Autoencoders**
(Ashuach et al., ACL 2026 — `assets/2026.acl-long.82.pdf`).

Paper code: <https://github.com/technion-cs-nlp/CRISP>

---

## TL;DR

Do **not** reproduce the paper as published (2400 fine-tuning runs behind a 200-config
Bayesian sweep — hundreds of GPU-hours, thousands of dollars). Instead reproduce the
**Gemma-2-2B rows of Table 1** using the fixed best hyperparameters from Appendix F.
That is a handful of sub-$2 runs and makes or breaks the paper's central comparison.
Llama-3.1-8B is out of scope: the weights, its SAEs and the larger activations do not
fit a Colab session. The Table 3 (Harry Potter) rows are out of scope too — the WMDP
domains are what the paper's safety claim rests on, and each extra corpus is storage
the free Drive tier does not have.

**Pass/fail target:** the four Gemma-2-2B rows of Table 1 (WMDP-Bio + WMDP-Cyber).

---

## Constraints that shape everything

| Constraint | Implication |
|---|---|
| Local machine is Apple M4, 24 GB unified memory, no CUDA | All training and evaluation runs on Google Colab (`notebooks/crisp_colab.ipynb`, L4 recommended); the laptop only fetches the datasets and runs the tests. |
| Released code is **partial** | `/crisp` has feature selection + LoRA optimization + eval; a runnable HP demo notebook. WMDP is **not** a turnkey script and the corpora need separate access. |
| Project pins Python 3.13 + `uv` | Paper ships `environment.yml` (conda) with an older torch/SAELens stack. **Drop to Python 3.11** — SAELens + torch pins are unlikely to resolve on 3.13. |
| Eval uses an LLM judge pinned to Claude Sonnet 4 `2025-05-14` | That version is no longer callable. Substitute a current model and **calibrate against the original-model baselines** — your Fluency/Concept numbers won't be directly comparable to the paper's otherwise. |
| WMDP MCQ val/test split is the authors' own | Expect 1–2 points of drift on unlearn/retain accuracy unless the repo fixes the seed. Not a reproduction failure. |

---

## Day-one actions

1. **Request the WMDP bio-forget corpus from CAIS now** — it lives in its own gated repo,
   [`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus),
   not in `cais/wmdp-corpora`, and the bio configs already point at it
   (`data.target_corpus_repo`). Approval is unpredictable and gates half of Table 1. The
   MCQs (`cais/wmdp`) and the cyber/retain corpora are open — build and test the eval half,
   and run the **cyber** pair, while waiting.

2. **Run `configs/smoke.yaml` end-to-end, then the cyber pair.** The smoke config
   exercises fetch → select → train → eval → report on a tiny random model in about a
   minute, and the cyber pair is the first run against real published numbers — its
   corpora are public and total ~80 MB. If cyber does not land near Table 1, nothing
   downstream is trustworthy.

---

## Environment

```bash
# Drop to 3.11 — do this before anything else
uv venv --python 3.11
source .venv/bin/activate

# Vendor the paper repo
git clone https://github.com/technion-cs-nlp/CRISP.git vendor/CRISP

# Install its stack (prefer their pins first, then patch for MPS/Py3.11)
pip install -r vendor/CRISP/requirements.txt
# Key deps: torch, transformers, sae_lens, peft (LoRA), datasets
```

SAE loading (Gemma Scope, via SAELens):

```python
from sae_lens import SAE
sae, cfg, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_14/width_16k/canonical",   # Bio/Cyber use layers [4,6,8,10,12,14]
)
```

---

## Scoped reproduction — Gemma-2-2B

Use the **fixed best hyperparameters from Appendix F** (no re-sweep):

| Setting | WMDP-Bio | WMDP-Cyber |
|---|---|---|
| SAE layers (feature selection) | [4,6,8,10,12,14] | [4,6,8,10,12,14] |
| LoRA edit layers | [3–9] (early) | [3–9] (early) |
| k (salient features) | 30 | 50 |
| λ (unlearn scale, Eq. 9) | 30 | 20 |
| LoRA rank | 8 | 4 |
| learning rate | 4e-5 | 4e-5 |
| τ (ρ threshold) | 3 | 3 |
| β (retain), γ (coherence), α (unlearn) | 0.99, 0.01, 0.01 | same |

Data prep (per paper §4.1): 5000 random target/retain docs for Bio, all 986 for Cyber;
strip markdown/citations/image links/non-ASCII; right-truncate each doc to 1000 characters.

**Cost per run:** one forward pass over ~5000 docs (a few M tokens, minutes) + a short
LoRA fine-tune ≈ **< 1 hour on an A100 40 GB (~$1.20/hr RunPod/Lambda)**. Reproducing
CRISP + RMU + ELM at best configs across both domains is **< $50 compute**.

Deliverable: the four Gemma-2-2B rows of Table 1.

---

## Evaluation (the harder half)

Six metrics → harmonic mean (Eq. 12): `HM(100−U, R, M, F·50, C·50)`.

- **U** (unlearn acc, ↓), **R** (retain acc), **M** (MMLU) — from held-out MCQ. Their
  val/test split is bespoke.
- **F** (fluency 0–2), **C** (concept 0–2) — from an LLM judge over 100 prefixes/domain,
  greedy decoding, max 50 tokens. Judge prompts are verbatim in **Tables 9 & 10**.

**Calibration protocol (mandatory):** score the *unedited* model's generations with your
substitute judge first and check you recover the paper's baselines —
Llama Bio 1.24/1.77, Gemma Bio 1.07/1.78. If your judge reads systematically high/low,
report the offset and compare **within your own run**, not against the paper's table.
Budget ~$5–10 in API calls.

**Metric-design caveat worth flagging in the writeup:** the overall score is a harmonic
mean, dominated by its smallest term. ELM's 0.25 fluency → 12.5 after ×50 scaling is what
tanks its overall to 33.93. The headline "5–34 point" gap may be substantially a
metric-design artifact — worth isolating.

---

## Zero-budget fallback

- **Kaggle:** ~30 free GPU-hrs/week on 2× T4 — holds Gemma-2-2B + SAEs comfortably;
  enough for the entire Gemma-2-2B track.
- **Colab Pro:** $10/mo.
- Checkpoint aggressively — free sessions die.

---

## Phasing

| Phase | Scope | Compute | Gate |
|---|---|---|---|
| 0 | Env + smoke config + cyber pair | M4 / free T4 | Cyber rows near Table 1 |
| 1 | Gemma-2-2B Table 1, both domains, 3 methods | < $50 A100 | Recover the four Gemma rows |

Once phase 1's pipeline runs, each **ablation** is one more sub-$2 fine-tuning run — see
`ABLATIONS.md` (to be written) for the planned set.

---

## Open reproduction hazards (log deviations as you hit them)

- Substitute judge ≠ Sonnet 4 `2025-05-14` → calibrate, don't compare raw.
- Bespoke MCQ split → ±1–2 pts drift expected.
- SAELens/torch pins vs. Python 3.11/MPS → expect to patch pins.
- WMDP corpus access latency → start with MCQs + HP + (if open) cyber corpus.
