"""Shared helpers: devices, dtypes, seeding, batching."""

from __future__ import annotations

import logging
import os
import random
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

T = TypeVar("T")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str | None, device: torch.device) -> torch.dtype:
    if dtype and dtype != "auto":
        return getattr(torch, dtype)
    # bf16 optimiser math is unreliable on MPS, so only CUDA defaults to bf16.
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def cycle(items: Sequence[T], rng: random.Random) -> Iterator[T]:
    """Infinite shuffled iterator over ``items``."""
    order = list(range(len(items)))
    while True:
        rng.shuffle(order)
        for idx in order:
            yield items[idx]


def sample_batches(
    items: Sequence[T], batch_size: int, steps: int, seed: int
) -> Iterator[list[T]]:
    rng = random.Random(seed)
    stream = cycle(items, rng)
    for _ in range(steps):
        yield [next(stream) for _ in range(batch_size)]


def harmonic_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if any(v <= 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    return None
