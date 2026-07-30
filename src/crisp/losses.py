"""CRISP training objectives (Section 3.3, Eq. 9-11)."""

from __future__ import annotations

import torch

from .sae import SAEBundle


def unlearning_loss(
    acts_by_layer: dict[int, torch.Tensor],
    token_mask: torch.Tensor,
    saes: SAEBundle,
    salient: dict[int, list[int]],
    lambda_scale: float,
) -> torch.Tensor:
    """Eq. 9.

    ``E_t E_{f_i in F_salient} [ a_i^(t) + lambda * c_t ]`` where ``c_t`` is the
    mean activation over *all* features at token ``t``. Computed per layer and
    averaged over the pre-selected layers.
    """
    per_layer = []
    for layer, hidden in acts_by_layer.items():
        features = salient.get(layer, [])
        if not features:
            continue
        flat = hidden[token_mask]  # [n_tokens, d_model]
        acts = saes[layer].encode(flat).float()  # [n_tokens, d_sae]
        index = torch.tensor(features, device=acts.device, dtype=torch.long)
        salient_mean = acts.index_select(1, index).mean(dim=1)  # [n_tokens]
        all_mean = acts.mean(dim=1)  # c_t
        per_layer.append((salient_mean + lambda_scale * all_mean).mean())

    if not per_layer:
        raise ValueError("no salient features selected on any layer")
    return torch.stack(per_layer).mean()


def representation_distance(
    acts: dict[int, torch.Tensor],
    ref_acts: dict[int, torch.Tensor],
    token_mask: torch.Tensor,
    reduction: str = "sqnorm",
) -> torch.Tensor:
    """Eq. 10: mean over tokens of ``||h_M - h_M0||_2^2``, averaged over layers.

    ``reduction='mse'`` divides by d_model, which rescales the retention term
    without changing its direction.
    """
    per_layer = []
    for layer, hidden in acts.items():
        diff = (hidden[token_mask].float() - ref_acts[layer][token_mask].float()) ** 2
        value = diff.sum(dim=-1).mean() if reduction == "sqnorm" else diff.mean()
        per_layer.append(value)
    return torch.stack(per_layer).mean()


def total_loss(
    unlearn: torch.Tensor,
    retain: torch.Tensor,
    coherence: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Eq. 11."""
    return alpha * unlearn + beta * retain + gamma * coherence
