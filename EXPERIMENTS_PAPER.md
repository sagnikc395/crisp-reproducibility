# Experiments in the CRISP Paper

Source: *CRISP: Persistent Concept Unlearning via Sparse Autoencoders* — Ashuach, Arad, Mueller, Tutek, Belinkov (ACL 2026, pp. 1806–1825). Local copy: `assets/2026.acl-long.82.pdf`. Code: github.com/technion-cs-nlp/CRISP.

This document records what the paper runs and what it reports, as the reference target for reproduction. It describes the paper in full, including rows this repo does not attempt.

**What this repo reproduces:** the Gemma-2-2B rows on WMDP-Bio and WMDP-Cyber (Experiments 1–3). The Llama-3.1-8B rows and the Harry Potter benchmark (Experiment 4) are out of scope — Llama-3.1-8B does not fit a Colab session, and each extra corpus costs storage the free Drive tier does not have. The tables below are kept intact as the reference; treat them as the paper's numbers, not as targets this repo checks against.

## Shared Setup

- **Models.** Llama-3.1-8B with Llama Scope SAEs; Gemma-2-2B with Gemma Scope SAEs. Suppression is applied at layer 24 (Llama) and layer 14 (Gemma); feature analysis additionally checks layers 20/22 (Llama) and 10/12 (Gemma).
- **Method.** Two phases — (1) select salient SAE features by contrasting target vs. retain corpora (top-k activation-count difference, filtered by relative activation ratio ≥ τ), (2) LoRA fine-tune with `L_total = α·L_unlearn + β·L_retain + γ·L_coherence`.
- **Baselines.** RMU, ELM, and the unedited "Original" model.
- **Metrics.** Unlearn accuracy (↓), retain accuracy (↑), MMLU (↑), Fluency (0–2), Concept (0–2), and Overall = `HM(100−U, R, M, F·50, C·50)` — harmonic mean, so it penalizes any weak axis.
- **Sweep.** 200 hyperparameter configurations per method. Best config chosen on the validation split by unlearning efficacy, retain accuracy, and MMLU (first 10 questions per subject).
- **Data handling.** WMDP-Bio: 5000 randomly sampled target and retain entries. WMDP-Cyber: all 986 entries. Documents are stripped of markdown headers, citations, image links, and non-ASCII characters, then right-truncated to 1000 characters. WMDP MCQs are split evenly into validation and test; the same splits are used for every method.

## Experiment 1 — WMDP-Bio / WMDP-Cyber Unlearning (Table 1)

The main quantitative result. Retention is measured on MMLU subsets: high school + college biology for Bio, high school + college computer science for Cyber. Coherence uses 20 auxiliary sentences per domain generated with Claude Sonnet 4.

| Setting | Method | Overall ↑ | Unlearn ↓ | Retain ↑ | MMLU ↑ | Fluency ↑ | Concept ↑ |
|---|---|---|---|---|---|---|---|
| Bio / Llama-3.1-8B | Original | 56.60 | 68.29 | 76.81 | 61.15 | 1.24 | 1.77 |
| | ELM | 33.93 | 41.44 | 62.17 | 55.31 | 0.25 | 1.24 |
| | RMU | 52.51 | 34.54 | 67.75 | 59.50 | 0.56 | **1.58** |
| | **CRISP** | **60.10** | **30.93** | **74.13** | **60.28** | **0.77** | **1.58** |
| Bio / Gemma-2-2B | Original | 54.37 | 55.26 | 55.27 | 46.30 | 1.07 | 1.78 |
| | ELM | 22.13 | 27.80 | 40.54 | 35.80 | 0.14 | 1.20 |
| | RMU | 51.91 | **27.79** | 48.77 | 42.77 | 0.76 | **1.63** |
| | **CRISP** | **56.70** | 29.67 | **54.45** | **46.33** | **0.92** | **1.63** |
| Cyber / Llama-3.1-8B | Original | 61.32 | 40.95 | 54.00 | 61.15 | 1.27 | 1.43 |
| | ELM | 58.91 | 30.78 | 53.00 | 58.56 | 0.99 | 1.40 |
| | RMU | 52.47 | 33.70 | **55.00** | **61.15** | 0.68 | 1.23 |
| | **CRISP** | **61.74** | **29.38** | 53.00 | 58.86 | **1.14** | **1.49** |
| Cyber / Gemma-2-2B | Original | 52.57 | 33.90 | 39.00 | 46.30 | 1.05 | 1.46 |
| | ELM | 43.33 | 28.87 | 29.00 | 38.71 | 0.76 | **1.36** |
| | RMU | 44.79 | 28.67 | 36.00 | 44.79 | 0.64 | 1.23 |
| | **CRISP** | **49.02** | **27.26** | **38.00** | **46.26** | **0.81** | 1.28 |

**Result.** CRISP has the best Overall score in all four settings. On WMDP-Bio it gains ~27 (Llama) and ~34 (Gemma) points over ELM, and 8 (Llama) / 5 (Gemma) points over RMU. Gaps on WMDP-Cyber are more modest. RMU and ELM each achieve slightly lower unlearn accuracy in one setting (Bio on Gemma-2-2B) but at a substantially larger cost to retention, MMLU, and fluency. Standard deviations for Fluency and Concept are in the paper's Table 11.

## Experiment 2 — Unlearn/Retain Trade-off Sweep (Figures 2 and 4)

All 200 configurations per method are plotted as retain accuracy vs. unlearn accuracy, isolating the trade-off from MMLU and generation quality. The "ideal unlearning" point is random guessing on the unlearn benchmark with unchanged retain accuracy.

**Result.** On WMDP-Bio, CRISP is Pareto-dominant for both models, with many configurations clustering near the ideal point. Among baselines RMU generally beats ELM. On WMDP-Cyber the advantage narrows: on Llama-3.1-8B all three methods achieve similar trade-offs; on Gemma-2-2B CRISP and RMU are comparable while ELM lags. Some configurations slightly exceed the original model's retain accuracy. The tight clustering of CRISP and RMU points near the ideal is read as robustness to hyperparameter choice.

## Experiment 3 — Generation Quality (Table 2, qualitative)

Both models are prompted on non-harmful continuations containing WMDP-Bio concepts ("Carcinogens increase the risk of cancer…", "Viral replication requires host cells…"). Fluency and Concept scores follow the AxBench framework: 100 prompts per domain covering both general-domain concepts ("genetics", "encryption") and target-specific concepts ("infection", "malware"), with prompt construction and evaluation done by Claude Sonnet 4.

**Result.** RMU collapses into repetition loops on in-domain content; ELM emits markup artifacts and drifts off-topic or refuses even on harmless prompts. CRISP continues fluently with appropriate domain terminology and no repetition. This matches the Fluency/Concept columns in Table 1, where CRISP has the best fluency in every setting.

## Experiment 4 — Harry Potter Benchmark (Appendix B, Table 3)

Versatility beyond safety domains, using the HP multiple-choice benchmark from ELM. No retain-accuracy column here.

| Model | Method | Overall ↑ | Unlearn ↓ | MMLU ↑ | Fluency ↑ | Concept ↑ |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | Original | 47.87 | 74.19 | 65.96 | 0.90 | 1.52 |
| | ELM | 34.82 | 32.74 | 58.35 | 0.26 | 1.14 |
| | **RMU** | **58.02** | 34.19 | **61.15** | **0.82** | **1.44** |
| | CRISP | 53.81 | **29.52** | 60.64 | 0.64 | 1.38 |
| Gemma-2-2B | Original | 44.29 | 63.06 | 48.94 | 0.64 | 1.46 |
| | ELM | 17.18 | 27.10 | 38.19 | 0.10 | 0.80 |
| | RMU | 41.59 | 29.68 | **45.15** | 0.42 | 1.42 |
| | **CRISP** | **49.30** | **25.65** | 44.77 | **0.68** | **1.44** |

**Result.** CRISP achieves the lowest unlearn accuracy on both models, and the best Overall on Gemma-2-2B. This is the one reported setting where a baseline takes the Overall win: RMU scores 58.02 vs. CRISP's 53.81 on Llama-3.1-8B, driven by higher fluency and MMLU.

## Experiment 5 — Feature Analysis (§6, Figure 3, Appendix C, Tables 4–7)

Selected SAE features in the biosecurity domain are bucketed by activation frequency into **Target** (salient on harmful data), **Benign** (salient on retain data), and **Shared** (frequent in both). Top-5 tokens by logit contribution and Neuronpedia explanations are reported per group.

**Result.**
- Target features are semantically coherent harmful-biosecurity concepts: viral pathogens (Llama 3745, Gemma 4623), disease transmission (Llama 19213, 25550; Gemma 15109), biological threat vectors (Llama 22405, Gemma 1814).
- Benign features cover anatomy, physiology, clinical protocols, and research methodology (Llama 11025, 25529, 2840; Gemma 3164, 11152) — evidence that non-harmful biology is preserved.
- Shared features are mostly formatting, structural tokens, and domain-neutral terminology (Llama 20547, 741; Gemma 579, 15887), which CRISP leaves alone.
- Two Gemma-2-2B features look misselected by their Neuronpedia labels — 4008 ("flower-related") and 11127 ("financial crisis") — but on inspection fire on viral genome replication and on poisoning/terrorism contexts respectively. The paper reads this as conceptual entanglement in the SAE or a limitation of token-level explanations, not a selection failure.
- Trends hold on additional layers (Llama 20/22, Gemma 10/12).

## Stated Limitations

1. Depends on pretrained SAEs; effectiveness degrades where SAEs are poorly trained or fail to disentangle features.
2. Evaluation is confined to safety-critical domains (plus Harry Potter); generalization to new tasks/domains is untested.
3. No formal guarantee of complete removal — residual information may persist in distributed representations, and robustness to adversarial extraction is left to future work.
