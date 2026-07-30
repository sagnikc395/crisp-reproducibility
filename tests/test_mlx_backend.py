"""MLX inference backend. Skipped unless mlx-lm is installed (macOS only)."""

from __future__ import annotations

import pytest

from crisp.config import Config

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from crisp.mlx_backend import MLXCausalLM, load_mlx_model_and_tokenizer, resolve_mlx_repo  # noqa: E402


class FakeTokenizer:
    """Character-level stand-in so tests need no downloads."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [2 + (ord(c) % 30) for c in text]


def tiny_model(vocab_size: int = 64, tie: bool = True):
    from mlx_lm.models.llama import Model, ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=vocab_size,
        tie_word_embeddings=tie,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("tie", [True, False])
def test_final_logits_matches_unpadded_forward(tie):
    """Right padding must not perturb the last real position (causal attention)."""
    model = tiny_model(tie=tie)
    wrapper = MLXCausalLM(model, FakeTokenizer(), "tiny")

    prompts = ["a" * 3, "b" * 11, "c" * 7]  # deliberately ragged
    batched = wrapper.final_logits(prompts, max_len=64)
    assert batched.shape == (3, 64)

    for i, prompt in enumerate(prompts):
        alone = wrapper.final_logits([prompt], max_len=64)[0]
        assert (batched[i] - alone).abs().max() < 1e-4


def test_final_logits_equals_full_forward_last_token():
    """The split backbone/head path agrees with calling the model end to end."""
    model = tiny_model()
    tokenizer = FakeTokenizer()
    wrapper = MLXCausalLM(model, tokenizer, "tiny")

    prompt = "hello world"
    ids = mx.array([tokenizer.encode(prompt)])
    reference = model(ids)[0, -1]
    mx.eval(reference)

    got = wrapper.final_logits([prompt], max_len=64)[0]
    assert (got - memoryview_to_torch(reference)).abs().max() < 1e-4


def memoryview_to_torch(arr):
    import numpy as np
    import torch

    return torch.from_numpy(np.array(arr.astype(mx.float32)))


def test_long_prompt_keeps_the_tail():
    """Truncation must not drop the "Answer:" the MCQ scorer reads."""
    wrapper = MLXCausalLM(tiny_model(), FakeTokenizer(), "tiny")
    ids = wrapper._encode("x" * 100, max_len=16)
    assert len(ids) == 16
    assert ids == FakeTokenizer().encode("x" * 100)[-16:]


def test_repo_resolution_and_backend_guards(tmp_path):
    cfg = Config()
    assert resolve_mlx_repo(cfg) == "mlx-community/gemma-2-2b-4bit"
    cfg.model.mlx_name = "mlx-community/custom-4bit"
    assert resolve_mlx_repo(cfg) == "mlx-community/custom-4bit"

    # A PEFT adapter must be rejected with a pointer at the torch backend.
    peft_dir = tmp_path / "adapter"
    peft_dir.mkdir()
    (peft_dir / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="PEFT adapter"):
        load_mlx_model_and_tokenizer(cfg, str(peft_dir))


def test_train_is_refused():
    wrapper = MLXCausalLM(tiny_model(), FakeTokenizer(), "tiny")
    assert wrapper.eval() is wrapper
    with pytest.raises(RuntimeError, match="inference-only"):
        wrapper.train()
