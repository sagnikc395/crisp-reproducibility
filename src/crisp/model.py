"""Model loading, residual-stream capture and LoRA attachment."""

from __future__ import annotations

import contextlib
import re
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .utils import get_logger, hf_token, resolve_device, resolve_dtype

log = get_logger(__name__)


def load_model_and_tokenizer(cfg: Config):
    device = resolve_device(cfg.model.device)
    dtype = resolve_dtype(cfg.model.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, token=hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, dtype=dtype, token=hf_token()
    )
    model.to(device)
    model.eval()
    log.info("loaded %s on %s (%s)", cfg.model.name, device, dtype)
    return model, tokenizer, device, dtype


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The transformer block list, unwrapping PEFT if needed."""
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    for container in (base, getattr(base, "model", None), getattr(base, "transformer", None)):
        if container is None:
            continue
        for attr in ("layers", "h", "blocks"):
            blocks = getattr(container, attr, None)
            if isinstance(blocks, nn.ModuleList):
                return blocks
    raise AttributeError("could not locate decoder layers on model")


def _block_output(output):
    return output[0] if isinstance(output, tuple) else output


class ResidualCapture:
    """Capture ``hook_resid_post`` (block output) for a set of layers.

    Used both for SAE feature extraction and for the retention/coherence losses,
    so hooks stay attached inside the autograd graph.
    """

    def __init__(self, model: nn.Module, layers: Iterable[int]) -> None:
        self.layers = sorted(set(layers))
        self._blocks = decoder_layers(model)
        self.acts: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            self.acts[layer] = _block_output(output)

        return hook

    def __enter__(self) -> "ResidualCapture":
        self.acts = {}
        for layer in self.layers:
            self._handles.append(self._blocks[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextlib.contextmanager
def base_model(model: nn.Module):
    """Run as the original frozen model M0 by disabling LoRA adapters.

    Avoids keeping a second copy of the weights in memory.
    """
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield model
    else:
        yield model


def lora_target_modules(model: nn.Module, layers: list[int], suffixes: list[str]) -> list[str]:
    """Full module names for ``suffixes`` inside the given transformer blocks."""
    wanted = set(layers)
    targets = []
    pattern = re.compile(r"\.layers\.(\d+)\.")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = pattern.search(name + ".")
        if not match or int(match.group(1)) not in wanted:
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            targets.append(name)
    if not targets:
        raise ValueError(
            f"no LoRA targets matched layers={layers} suffixes={suffixes}"
        )
    return targets


def attach_lora(model: nn.Module, cfg: Config):
    from peft import LoraConfig, get_peft_model

    targets = lora_target_modules(model, cfg.model.lora_layers, cfg.model.lora_targets)
    lora_cfg = LoraConfig(
        r=cfg.train.lora_rank,
        lora_alpha=cfg.train.resolved_lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    log.info("attached LoRA to %d modules (%d trainable params)", len(targets), trainable)
    return peft_model


def tokenize_batch(tokenizer, texts: list[str], max_len: int, device: torch.device):
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    return {k: v.to(device) for k, v in batch.items()}
