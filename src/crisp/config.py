"""Experiment configuration.

Field defaults follow Appendix F of the CRISP paper; per-model/per-domain
overrides live in ``configs/*.yaml``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "google/gemma-2-2b"
    device: str = "auto"
    dtype: str = "auto"
    #: ``torch`` | ``mlx``. The MLX backend is inference-only (``crisp eval``);
    #: selection and training always run on torch.
    backend: str = "torch"
    #: MLX checkpoint to evaluate; defaults to a known mlx-community mirror of
    #: ``name`` (see ``mlx_backend.DEFAULT_MLX_REPOS``).
    mlx_name: str | None = None
    # Layers whose residual stream is read by an SAE and suppressed (Appendix F).
    sae_layers: list[int] = field(default_factory=lambda: [4, 6, 8, 10, 12, 14])
    # Transformer blocks that receive LoRA adapters ("early optimization layers", [3-9]).
    lora_layers: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7, 8, 9])
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class SAEConfig:
    #: ``gemma_scope`` | ``llama_scope`` | ``sae_lens`` | ``random`` (smoke tests)
    source: str = "gemma_scope"
    repo_id: str = "google/gemma-scope-2b-pt-res"
    #: Format string resolved with ``layer``/``width``. ``auto`` lists the repo and
    #: picks the released sparsity closest to ``l0_target`` (Gemma Scope publishes
    #: several ``average_l0_*`` variants per layer).
    filename_template: str = "auto"
    #: Target average L0; 100 matches the Gemma Scope "canonical" selection rule.
    l0_target: int = 100
    #: llama-scope publishes one repo per layer instead of one file per layer.
    repo_template: str | None = None
    width: int = 16384
    d_model: int | None = None


@dataclass
class DataConfig:
    domain: str = "bio"  # bio | cyber
    target_corpus: str | None = None  # HF name or local .jsonl/.txt path
    retain_corpus: str | None = None
    max_target_docs: int = 5000
    max_retain_docs: int = 5000
    max_chars: int = 1000
    max_seq_len: int = 256
    coherence_path: str | None = None
    seed: int = 0


@dataclass
class SelectionConfig:
    top_k: int = 30  # k in top-k(F, Delta phi)
    tau: float = 3.0  # relative activation ratio threshold
    epsilon: float = 1e-6
    max_docs: int = 500  # docs per corpus used for the activation statistics
    batch_size: int = 4
    cache_dir: str = "outputs/features"


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 2
    lr: float = 4e-5
    lora_rank: int = 8
    lora_alpha: int | None = None  # defaults to 2 * rank
    lora_dropout: float = 0.0
    grad_clip: float = 1.0
    lambda_scale: float = 30.0  # lambda in Eq. 9 (mean-activation penalty)
    beta: float = 0.99  # retention weight
    gamma: float = 0.01  # coherence weight
    alpha: float | None = None  # unlearning weight; defaults to 1 - beta
    #: ``sqnorm`` follows Eq. 10 literally; ``mse`` additionally divides by d_model.
    retain_reduction: str = "sqnorm"
    warmup_steps: int = 0
    log_every: int = 10
    save_dir: str = "outputs/runs"
    seed: int = 0

    @property
    def resolved_alpha(self) -> float:
        return self.alpha if self.alpha is not None else 1.0 - self.beta

    @property
    def resolved_lora_alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_rank


@dataclass
class EvalConfig:
    split: str = "test"  # MCQs are halved into validation / test
    mcq_batch_size: int = 4
    mmlu_subjects: list[str] = field(default_factory=list)
    mmlu_max_per_subject: int = 0  # 0 -> all
    gen_prompts_path: str | None = None
    gen_max_new_tokens: int = 50
    judge_model: str = "claude-sonnet-4-5"
    judge_concept: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "crisp"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        kwargs: dict[str, Any] = {}
        sections = {f.name: f.type for f in dataclasses.fields(cls)}
        for key, value in raw.items():
            if key not in sections:
                raise ValueError(f"unknown config section: {key}")
            if key == "run_name":
                kwargs[key] = value
                continue
            section_cls = {
                "model": ModelConfig,
                "sae": SAEConfig,
                "data": DataConfig,
                "selection": SelectionConfig,
                "train": TrainConfig,
                "eval": EvalConfig,
            }[key]
            known = {f.name for f in dataclasses.fields(section_cls)}
            unknown = set(value or {}) - known
            if unknown:
                raise ValueError(f"unknown keys in [{key}]: {sorted(unknown)}")
            kwargs[key] = section_cls(**(value or {}))
        return cls(**kwargs)

    def apply_overrides(self, overrides: list[str]) -> "Config":
        """Apply ``section.field=value`` CLI overrides in place."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got {item!r}")
            key, raw_value = item.split("=", 1)
            section_name, _, field_name = key.partition(".")
            if not field_name:
                setattr(self, section_name, yaml.safe_load(raw_value))
                continue
            section = getattr(self, section_name)
            if not hasattr(section, field_name):
                raise ValueError(f"unknown override target: {key}")
            setattr(section, field_name, yaml.safe_load(raw_value))
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
