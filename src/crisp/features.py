"""Contrastive SAE feature selection (Section 3.2, Eq. 3-8)."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .model import ResidualCapture, tokenize_batch
from .sae import SAEBundle
from .utils import chunked, get_logger

log = get_logger(__name__)


@dataclass
class LayerStats:
    """Per-layer activation statistics over one corpus."""

    count: torch.Tensor  # phi(f_i, D): tokens with non-zero activation  [d_sae]
    total: torch.Tensor  # A(f_i, D): cumulative activation magnitude    [d_sae]
    n_tokens: int


@torch.no_grad()
def corpus_statistics(
    model,
    tokenizer,
    saes: SAEBundle,
    docs: list[str],
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    desc: str = "stats",
) -> dict[int, LayerStats]:
    """Accumulate phi (Eq. 3) and A (Eq. 5) for every SAE feature and layer.

    The accumulators live on the CPU in float64: MPS has no float64 at all, and
    summing hundreds of thousands of token activations in float32 loses counts
    once the running total is large. Only a ``[d_sae]`` vector crosses the
    device boundary per batch, so the transfer is negligible next to the forward
    pass.
    """
    stats = {
        layer: LayerStats(
            count=torch.zeros(saes[layer].d_sae, dtype=torch.float64),
            total=torch.zeros(saes[layer].d_sae, dtype=torch.float64),
            n_tokens=0,
        )
        for layer in saes.layers
    }

    for batch_docs in tqdm(list(chunked(docs, batch_size)), desc=desc, leave=False):
        batch = tokenize_batch(tokenizer, batch_docs, max_seq_len, device)
        mask = batch["attention_mask"].bool()
        with ResidualCapture(model, saes.layers) as capture:
            model(**batch)
            for layer in saes.layers:
                hidden = capture.acts[layer][mask]  # [n_tokens, d_model]
                acts = saes[layer].encode(hidden).float()
                stats[layer].count += (acts > 0).sum(dim=0).cpu().double()
                stats[layer].total += acts.sum(dim=0).cpu().double()
                stats[layer].n_tokens += hidden.shape[0]
                del acts, hidden
    return stats


@dataclass
class SelectedFeatures:
    """Salient target features per layer, plus diagnostics."""

    layers: dict[int, list[int]]
    scores: dict[int, dict[str, list[float]]]
    top_k: int
    tau: float

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "top_k": self.top_k,
            "tau": self.tau,
            "layers": {str(k): v for k, v in self.layers.items()},
            "scores": {str(k): v for k, v in self.scores.items()},
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SelectedFeatures":
        raw = json.loads(Path(path).read_text())
        return cls(
            layers={int(k): v for k, v in raw["layers"].items()},
            scores={int(k): v for k, v in raw.get("scores", {}).items()},
            top_k=raw["top_k"],
            tau=raw["tau"],
        )

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.layers.values())


def select_features(
    target_stats: dict[int, LayerStats],
    retain_stats: dict[int, LayerStats],
    top_k: int,
    tau: float,
    epsilon: float = 1e-6,
) -> SelectedFeatures:
    """top-k by activation-count difference (Eq. 4, 7), filtered by relative
    activation ratio >= tau (Eq. 6, 8)."""
    layers: dict[int, list[int]] = {}
    scores: dict[int, dict[str, list[float]]] = {}

    for layer in target_stats:
        tgt, ret = target_stats[layer], retain_stats[layer]
        # Normalise counts by corpus size so unequal corpora stay comparable.
        scale = tgt.n_tokens / max(ret.n_tokens, 1)
        delta_phi = tgt.count - ret.count * scale
        rho = tgt.total / (ret.total * scale + epsilon)

        k = min(top_k, delta_phi.numel())
        freq_idx = torch.topk(delta_phi, k).indices
        keep = freq_idx[rho[freq_idx] >= tau]

        layers[layer] = sorted(int(i) for i in keep.tolist())
        scores[layer] = {
            "feature": [int(i) for i in freq_idx.tolist()],
            "delta_phi": [float(delta_phi[i]) for i in freq_idx.tolist()],
            "rho": [float(rho[i]) for i in freq_idx.tolist()],
        }
        log.info(
            "layer %2d: %d/%d top-k features pass tau=%.1f",
            layer,
            len(layers[layer]),
            k,
            tau,
        )
    return SelectedFeatures(layers=layers, scores=scores, top_k=top_k, tau=tau)


def run_selection(
    model,
    tokenizer,
    saes: SAEBundle,
    target_docs: list[str],
    retain_docs: list[str],
    cfg,
    device: torch.device,
) -> SelectedFeatures:
    sel = cfg.selection
    target_docs = target_docs[: sel.max_docs] if sel.max_docs else target_docs
    retain_docs = retain_docs[: sel.max_docs] if sel.max_docs else retain_docs

    target_stats = corpus_statistics(
        model, tokenizer, saes, target_docs, cfg.data.max_seq_len,
        sel.batch_size, device, desc="target stats",
    )
    retain_stats = corpus_statistics(
        model, tokenizer, saes, retain_docs, cfg.data.max_seq_len,
        sel.batch_size, device, desc="retain stats",
    )
    return select_features(target_stats, retain_stats, sel.top_k, sel.tau, sel.epsilon)
