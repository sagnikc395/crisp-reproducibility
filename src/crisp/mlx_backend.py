"""MLX-LM inference backend for Apple silicon.

Evaluation only. CRISP training needs gradients through the residual stream and
per-layer hooks, which mlx-lm's forward pass does not expose, so ``select``,
``train`` and ``baseline`` stay on the torch/PEFT path. What ``eval`` asks of a
model is narrower — score the four answer letters of an MCQ, and greedily
continue a prefix — and both map cleanly onto mlx-lm.

Two details worth knowing:

* MCQ batches are **right**-padded here, not left-padded as in the torch path.
  mlx-lm builds its own causal mask and takes no attention mask, so a padding
  mask cannot be supplied; with causal attention a trailing pad cannot affect
  the hidden state at an earlier real token, so right padding is exact while
  left padding would leak pad positions into every query.
* The logits for a ``[B, L]`` batch are ``[B, L, 256k]`` on Gemma — several GB.
  Only the final real position is ever needed, so the backbone and the LM head
  are called separately and the head sees one vector per sequence.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .utils import get_logger

log = get_logger(__name__)

#: Convenience defaults so a config can keep naming the original HF checkpoint.
DEFAULT_MLX_REPOS = {
    "google/gemma-2-2b": "mlx-community/gemma-2-2b-4bit",
    "google/gemma-2-2b-it": "mlx-community/gemma-2-2b-it-4bit",
    "meta-llama/Llama-3.1-8B": "mlx-community/Meta-Llama-3.1-8B-4bit",
    "meta-llama/Llama-3.1-8B-Instruct": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}


def resolve_mlx_repo(cfg: Config) -> str:
    return cfg.model.mlx_name or DEFAULT_MLX_REPOS.get(cfg.model.name, cfg.model.name)


class MLXCausalLM:
    """Duck-types the slice of the HF model API that ``crisp eval`` uses.

    ``eval_mcq`` and ``eval_gen`` dispatch on ``final_logits`` / ``generate_texts``
    being present, so nothing else in the pipeline needs to know the backend.
    """

    def __init__(self, model, tokenizer, name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.name = name

    # -- helpers ---------------------------------------------------------
    def _encode(self, text: str, max_len: int) -> list[int]:
        ids = self.tokenizer.encode(text)
        # Keep the *tail*: an MCQ prompt ends in "Answer:", which is the only
        # position we read. (The torch path truncates the tail instead, but
        # neither triggers at the default max_len of 1024.)
        return ids[-max_len:] if max_len and len(ids) > max_len else ids

    def _head(self, h):
        """Apply the LM head to ``[..., d_model]`` hidden states."""
        import mlx.core as mx

        model = self.model
        if getattr(model, "lm_head", None) is not None:
            out = model.lm_head(h)
        elif hasattr(getattr(model, "model", None), "embed_tokens"):
            out = model.model.embed_tokens.as_linear(h)  # tied embeddings
        else:
            raise AttributeError(
                f"cannot locate the LM head on {type(model).__name__}; "
                "the MLX backend supports tied-embedding and lm_head models"
            )
        cap = getattr(model, "final_logit_softcapping", None)
        if cap:  # Gemma-2 caps its logits; monotone, but keep parity with HF.
            out = mx.tanh(out / cap) * cap
        return out

    # -- the API eval uses -----------------------------------------------
    def final_logits(self, prompts: list[str], max_len: int = 1024) -> torch.Tensor:
        """Next-token logits at the last real token of each prompt, ``[B, vocab]``."""
        import mlx.core as mx
        import numpy as np

        ids = [self._encode(p, max_len) for p in prompts]
        lengths = [len(t) for t in ids]
        width = max(lengths)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id or 0
        tokens = mx.array([t + [pad] * (width - len(t)) for t in ids])

        hidden = self.model.model(tokens)  # [B, L, d_model]
        last = hidden[mx.arange(len(ids)), mx.array([n - 1 for n in lengths])]
        logits = self._head(last).astype(mx.float32)
        mx.eval(logits)
        return torch.from_numpy(np.array(logits))

    def generate_texts(self, prefixes: list[str], max_new_tokens: int = 50) -> list[str]:
        """Greedy continuations (Appendix E.2), batched via mlx-lm."""
        from mlx_lm.generate import batch_generate

        prompts = [self.tokenizer.encode(p) for p in prefixes]
        # sampler=None in mlx-lm means argmax, i.e. greedy decoding.
        response = batch_generate(
            self.model, self.tokenizer, prompts, max_tokens=max_new_tokens
        )
        return [text.strip() for text in response.texts]

    # -- inert stand-ins so shared eval code keeps working ----------------
    def eval(self) -> "MLXCausalLM":
        return self

    def train(self, mode: bool = True) -> "MLXCausalLM":
        if mode:
            raise RuntimeError("the MLX backend is inference-only; train with backend: torch")
        return self


def _check_adapter(adapter: str | None) -> str | None:
    """Reject torch/PEFT adapters, which mlx-lm cannot read."""
    if adapter is None:
        return None
    path = Path(adapter)
    if (path / "adapters.safetensors").exists():
        return str(path)
    if (path / "adapter_model.safetensors").exists() or (path / "adapter_config.json").exists():
        raise ValueError(
            f"{adapter} is a PEFT adapter; the MLX backend can only load mlx-lm "
            "adapters (adapters.safetensors). Evaluate trained adapters with "
            "-o model.backend=torch."
        )
    raise FileNotFoundError(f"no adapter weights found in {adapter}")


def load_mlx_model_and_tokenizer(cfg: Config, adapter: str | None = None):
    """Mirror of ``model.load_model_and_tokenizer`` for the MLX backend."""
    try:
        from mlx_lm import load
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "the mlx backend needs mlx-lm: uv sync --extra mlx"
        ) from exc

    repo = resolve_mlx_repo(cfg)
    model, tokenizer = load(repo, adapter_path=_check_adapter(adapter))
    log.info("loaded %s via mlx-lm", repo)
    # device/dtype are returned for signature parity; MLX manages both itself.
    return MLXCausalLM(model, tokenizer, repo), tokenizer, torch.device("cpu"), torch.float32
