"""RMU baseline (Li et al., 2024 -- "Representation Misdirection for Unlearning").

Forget activations at a chosen layer are pushed towards ``steering * u`` for a
fixed random unit vector ``u``; retain activations are held at their original
values. Only the ``down_proj`` matrices of a small window of layers are updated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from ..model import ResidualCapture, tokenize_batch
from ..utils import get_logger, sample_batches, set_seed

log = get_logger(__name__)


@dataclass
class RMUConfig:
    layer_ids: list[int] = field(default_factory=lambda: [5, 6, 7])
    steering: float = 100.0  # steering coefficient c
    alpha: float = 50.0  # retain weight
    lr: float = 5.43e-5
    steps: int = 200
    batch_size: int = 2
    max_seq_len: int = 256
    param_suffix: str = "down_proj"
    seed: int = 0

    @property
    def read_layer(self) -> int:
        return max(self.layer_ids)


def resolve_layer_ids(layer_ids: list[int], n_layers: int) -> list[int]:
    """Fit RMU's update window inside a model that has fewer layers than the 7B
    models the defaults were tuned on (the smoke model has two)."""
    valid = sorted({i for i in layer_ids if 0 <= i < n_layers})
    if len(valid) == len(set(layer_ids)):
        return valid
    width = min(len(set(layer_ids)), n_layers)
    shifted = list(range(n_layers - width, n_layers))
    log.warning("RMU layers %s do not fit a %d-layer model; using %s",
                sorted(set(layer_ids)), n_layers, shifted)
    return shifted


def _trainable_params(model, layer_ids: list[int], suffix: str) -> list[torch.nn.Parameter]:
    wanted = {f".layers.{i}." for i in layer_ids}
    params = []
    for name, param in model.named_parameters():
        if any(w in name for w in wanted) and suffix in name and name.endswith("weight"):
            param.requires_grad_(True)
            params.append(param)
        else:
            param.requires_grad_(False)
    if not params:
        raise ValueError(f"no RMU parameters matched layers={layer_ids} suffix={suffix!r}")
    return params


def train_rmu(
    model,
    tokenizer,
    target_docs: list[str],
    retain_docs: list[str],
    cfg: RMUConfig,
    device: torch.device,
) -> dict:
    set_seed(cfg.seed)
    d_model = model.config.hidden_size
    cfg.layer_ids = resolve_layer_ids(cfg.layer_ids, model.config.num_hidden_layers)

    # Fixed random unit control vector, scaled by the steering coefficient.
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    control = torch.rand(d_model, generator=generator)
    control = control / control.norm() * cfg.steering
    control = control.to(device=device, dtype=torch.float32)

    params = _trainable_params(model, cfg.layer_ids, cfg.param_suffix)
    optimizer = AdamW(params, lr=cfg.lr)

    # Frozen reference activations require a second forward pass with the
    # original weights, so snapshot the trainable tensors before training.
    frozen = [p.detach().clone() for p in params]

    layer = cfg.read_layer
    target_batches = sample_batches(target_docs, cfg.batch_size, cfg.steps, cfg.seed)
    retain_batches = sample_batches(retain_docs, cfg.batch_size, cfg.steps, cfg.seed + 1)
    history = []
    model.train()
    start = time.time()

    for step in tqdm(range(cfg.steps), desc="rmu"):
        optimizer.zero_grad(set_to_none=True)

        target_batch = tokenize_batch(tokenizer, next(target_batches), cfg.max_seq_len, device)
        retain_batch = tokenize_batch(tokenizer, next(retain_batches), cfg.max_seq_len, device)
        target_mask = target_batch["attention_mask"].bool()
        retain_mask = retain_batch["attention_mask"].bool()

        # Reference activations of the original model. The weights are swapped
        # in place, so this must happen before any autograd graph is built.
        with torch.no_grad():
            current = [p.detach().clone() for p in params]
            for p, f in zip(params, frozen):
                p.copy_(f)
            with ResidualCapture(model, [layer]) as capture:
                model(**retain_batch)
                ref = capture.acts[layer][retain_mask].float().detach()
            for p, c in zip(params, current):
                p.copy_(c)
            del current

        with ResidualCapture(model, [layer]) as capture:
            model(**target_batch)
            acts = capture.acts[layer][target_mask].float()
        l_forget = ((acts - control) ** 2).mean()

        with ResidualCapture(model, [layer]) as capture:
            model(**retain_batch)
            acts = capture.acts[layer][retain_mask].float()
        l_retain = ((acts - ref) ** 2).mean()

        loss = l_forget + cfg.alpha * l_retain
        loss.backward()
        optimizer.step()

        history.append(
            {"step": step, "loss": float(loss.detach()),
             "forget": float(l_forget.detach()), "retain": float(l_retain.detach())}
        )
        if (step + 1) % 20 == 0:
            log.info("rmu step %d | loss %.4f | forget %.4f | retain %.4f",
                     step + 1, history[-1]["loss"], history[-1]["forget"], history[-1]["retain"])

    model.eval()
    return {"history": history, "seconds": time.time() - start}
