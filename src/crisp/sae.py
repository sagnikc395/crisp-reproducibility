"""Sparse autoencoders over the residual stream (Section 3.1, Eq. 1).

Supports pretrained Gemma Scope (JumpReLU) and Llama Scope SAEs, an optional
``sae_lens`` backend, and a random SAE for pipeline smoke tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import SAEConfig
from .utils import get_logger, hf_token

log = get_logger(__name__)


class SparseAutoencoder(nn.Module):
    """a(h) = sigma(W_enc h + b_enc);  h_hat(a) = W_dec a + b_dec."""

    def __init__(
        self,
        W_enc: torch.Tensor,  # [d_model, d_sae]
        b_enc: torch.Tensor,  # [d_sae]
        W_dec: torch.Tensor,  # [d_sae, d_model]
        b_dec: torch.Tensor,  # [d_model]
        activation: str = "jumprelu",
        threshold: torch.Tensor | None = None,
        k: int | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("W_dec", W_dec)
        self.register_buffer("b_dec", b_dec)
        self.register_buffer(
            "threshold", threshold if threshold is not None else torch.zeros_like(b_enc)
        )
        self.activation = activation
        self.k = k
        self.d_model = W_enc.shape[0]
        self.d_sae = W_enc.shape[1]

    def pre_acts(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.b_dec if self.activation == "topk_centered" else h) @ self.W_enc + self.b_enc

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Feature activations for residual-stream states ``h`` [..., d_model]."""
        pre = self.pre_acts(h.to(self.W_enc.dtype))
        if self.activation == "relu":
            return torch.relu(pre)
        if self.activation.startswith("topk"):
            k = self.k or 64
            values, indices = torch.topk(pre, k, dim=-1)
            out = torch.zeros_like(pre)
            return out.scatter(-1, indices, torch.relu(values))
        # JumpReLU (Gemma Scope): zero out anything below the learned threshold.
        return pre * (pre > self.threshold)

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_dec + self.b_dec

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(h))


@dataclass
class SAEBundle:
    """One SAE per suppressed layer."""

    saes: dict[int, SparseAutoencoder]

    def to(self, device: torch.device, dtype: torch.dtype) -> "SAEBundle":
        for sae in self.saes.values():
            sae.to(device=device, dtype=dtype)
        return self

    @property
    def layers(self) -> list[int]:
        return sorted(self.saes)

    def __getitem__(self, layer: int) -> SparseAutoencoder:
        return self.saes[layer]


_L0_RE = re.compile(r"average_l0_(\d+)")


def resolve_gemma_scope_filename(cfg: SAEConfig, layer: int) -> str:
    """Pick the released SAE for ``layer`` whose average L0 is nearest the target."""
    if cfg.filename_template != "auto":
        return cfg.filename_template.format(layer=layer, width=cfg.width)

    from huggingface_hub import list_repo_files

    width = f"width_{cfg.width // 1024}k"
    prefix = f"layer_{layer}/{width}/"
    candidates = [
        f
        for f in list_repo_files(cfg.repo_id, token=hf_token())
        if f.startswith(prefix) and f.endswith("params.npz")
    ]
    if not candidates:
        raise FileNotFoundError(f"no SAE for {prefix} in {cfg.repo_id}")

    def distance(name: str) -> tuple[int, int]:
        match = _L0_RE.search(name)
        # Prefer a canonical release if one exists, else nearest L0.
        return (0, 0) if "canonical" in name else (1, abs(int(match.group(1)) - cfg.l0_target))

    return min(candidates, key=distance)


def _load_gemma_scope(cfg: SAEConfig, layer: int) -> SparseAutoencoder:
    import numpy as np
    from huggingface_hub import hf_hub_download

    filename = resolve_gemma_scope_filename(cfg, layer)
    log.info("gemma-scope layer %d -> %s", layer, filename)
    path = hf_hub_download(repo_id=cfg.repo_id, filename=filename, token=hf_token())
    params = np.load(path)
    tensors = {k: torch.from_numpy(params[k]).float() for k in params.files}
    return SparseAutoencoder(
        W_enc=tensors["W_enc"],
        b_enc=tensors["b_enc"],
        W_dec=tensors["W_dec"],
        b_dec=tensors["b_dec"],
        activation="jumprelu",
        threshold=tensors.get("threshold"),
    )


_KEY_ALIASES = {
    "W_enc": ["W_enc", "encoder.weight", "sae.encoder.weight", "enc.weight"],
    "b_enc": ["b_enc", "encoder.bias", "sae.encoder.bias", "enc.bias"],
    "W_dec": ["W_dec", "decoder.weight", "sae.decoder.weight", "dec.weight"],
    "b_dec": ["b_dec", "decoder.bias", "sae.decoder.bias", "dec.bias", "b_D"],
    "threshold": ["threshold", "log_jumprelu_threshold", "jumprelu_threshold"],
}


def _pick(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor | None:
    for alias in _KEY_ALIASES[name]:
        if alias in tensors:
            return tensors[alias]
    return None


def _load_llama_scope(cfg: SAEConfig, layer: int, d_model: int) -> SparseAutoencoder:
    from huggingface_hub import list_repo_files, hf_hub_download
    from safetensors.torch import load_file

    repo_id = (cfg.repo_template or "fnlp/Llama3_1-8B-Base-L{layer}R-8x").format(layer=layer)
    files = [f for f in list_repo_files(repo_id, token=hf_token()) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"no .safetensors weights in {repo_id}")
    files.sort(key=len)
    path = hf_hub_download(repo_id=repo_id, filename=files[0], token=hf_token())
    raw = {k: v.float() for k, v in load_file(path).items()}

    W_enc = _pick(raw, "W_enc")
    W_dec = _pick(raw, "W_dec")
    if W_enc is None or W_dec is None:
        raise KeyError(f"unrecognised SAE key layout in {repo_id}: {sorted(raw)}")
    # Normalise orientation to [d_model, d_sae] / [d_sae, d_model].
    if W_enc.shape[0] != d_model:
        W_enc = W_enc.T
    if W_dec.shape[1] != d_model:
        W_dec = W_dec.T
    d_sae = W_enc.shape[1]

    b_enc = _pick(raw, "b_enc")
    b_dec = _pick(raw, "b_dec")
    threshold = _pick(raw, "threshold")
    if threshold is not None and "log_jumprelu_threshold" in raw:
        threshold = threshold.exp()

    return SparseAutoencoder(
        W_enc=W_enc.contiguous(),
        b_enc=b_enc if b_enc is not None else torch.zeros(d_sae),
        W_dec=W_dec.contiguous(),
        b_dec=b_dec if b_dec is not None else torch.zeros(d_model),
        activation="jumprelu" if threshold is not None else "relu",
        threshold=threshold,
    )


def _load_sae_lens(cfg: SAEConfig, layer: int) -> SparseAutoencoder:
    from sae_lens import SAE

    sae, _, _ = SAE.from_pretrained(
        release=cfg.repo_id, sae_id=cfg.filename_template.format(layer=layer)
    )
    state = sae.state_dict()
    threshold = state.get("threshold")
    activation = getattr(sae.cfg, "activation_fn_str", "relu")
    return SparseAutoencoder(
        W_enc=state["W_enc"].float(),
        b_enc=state["b_enc"].float(),
        W_dec=state["W_dec"].float(),
        b_dec=state["b_dec"].float(),
        activation="jumprelu" if threshold is not None else activation,
        threshold=threshold.float() if threshold is not None else None,
        k=getattr(sae.cfg, "k", None),
    )


def _random_sae(d_model: int, width: int, seed: int = 0) -> SparseAutoencoder:
    gen = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(d_model, width, generator=gen) / d_model**0.5
    return SparseAutoencoder(
        W_enc=W_enc,
        b_enc=torch.zeros(width),
        W_dec=W_enc.T.contiguous(),
        b_dec=torch.zeros(d_model),
        activation="relu",
    )


def load_saes(
    cfg: SAEConfig,
    layers: list[int],
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SAEBundle:
    saes: dict[int, SparseAutoencoder] = {}
    for layer in layers:
        if cfg.source == "gemma_scope":
            sae = _load_gemma_scope(cfg, layer)
        elif cfg.source == "llama_scope":
            sae = _load_llama_scope(cfg, layer, d_model)
        elif cfg.source == "sae_lens":
            sae = _load_sae_lens(cfg, layer)
        elif cfg.source == "random":
            sae = _random_sae(d_model, cfg.width, seed=layer)
        else:
            raise ValueError(f"unknown SAE source: {cfg.source}")
        if sae.d_model != d_model:
            raise ValueError(
                f"SAE for layer {layer} has d_model={sae.d_model}, model has {d_model}"
            )
        sae.requires_grad_(False)
        sae.eval()
        saes[layer] = sae
        log.info("loaded SAE layer %d: d_sae=%d (%s)", layer, sae.d_sae, sae.activation)
    return SAEBundle(saes).to(device, dtype)
