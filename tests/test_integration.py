"""End-to-end pipeline checks on a tiny random Llama (no gated downloads).

Skipped automatically when the tiny model cannot be fetched from the Hub.
"""

from __future__ import annotations

import pytest
import torch

from crisp.config import Config
from crisp.data import MCQItem
from crisp.eval_gen import generate_continuations
from crisp.eval_mcq import evaluate_mcq
from crisp.features import run_selection
from crisp.model import ResidualCapture, attach_lora, base_model, decoder_layers
from crisp.model import load_model_and_tokenizer, lora_target_modules
from crisp.sae import load_saes
from crisp.train import train_crisp

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def loaded():
    cfg = Config.from_yaml("configs/smoke.yaml")
    try:
        model, tokenizer, device, dtype = load_model_and_tokenizer(cfg)
    except Exception as exc:  # offline / hub failure
        pytest.skip(f"tiny model unavailable: {exc}")
    return cfg, model, tokenizer, device, dtype


def test_residual_capture_matches_hidden_states(loaded):
    cfg, model, tokenizer, device, _ = loaded
    batch = tokenizer("a short sentence for testing", return_tensors="pt").to(device)
    n_layers = model.config.num_hidden_layers
    with ResidualCapture(model, list(range(n_layers))) as capture:
        out = model(**batch, output_hidden_states=True)

    # For non-final blocks, hidden_states[i + 1] is exactly the block output --
    # the `hook_resid_post` site the pretrained SAEs were trained on.
    for layer in range(n_layers - 1):
        assert torch.allclose(capture.acts[layer], out.hidden_states[layer + 1], atol=1e-4)

    # HF applies the final RMSNorm to the last hidden_states entry, so the hook
    # (correctly) sees the un-normalised residual stream instead.
    last = capture.acts[n_layers - 1]
    normed = model.model.norm(last)
    assert torch.allclose(normed, out.hidden_states[-1], atol=1e-4)


def test_base_model_context_restores_original_outputs(loaded):
    cfg, model, tokenizer, device, _ = loaded
    batch = tokenizer("reference activations", return_tensors="pt").to(device)
    with torch.no_grad():
        reference = model(**batch).logits.clone()

    peft_model = attach_lora(model, cfg)
    # Perturb the adapter so enabled/disabled outputs must differ.
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.5)
    with torch.no_grad():
        adapted = peft_model(**batch).logits
        with base_model(peft_model):
            disabled = peft_model(**batch).logits
    assert torch.allclose(disabled, reference, atol=1e-4)
    assert not torch.allclose(adapted, reference, atol=1e-3)
    peft_model.unload()


def test_lora_targets_restricted_to_requested_layers(loaded):
    cfg, model, *_ = loaded
    targets = lora_target_modules(model, [1], ["q_proj", "down_proj"])
    assert targets and all(".layers.1." in t for t in targets)
    assert all(t.endswith(("q_proj", "down_proj")) for t in targets)
    with pytest.raises(ValueError):
        lora_target_modules(model, [999], ["q_proj"])


def test_mcq_evaluation_returns_a_valid_accuracy(loaded):
    cfg, model, tokenizer, device, _ = loaded
    items = [MCQItem(f"Question {i}?", ["alpha", "beta", "gamma", "delta"], i % 4) for i in range(8)]
    result = evaluate_mcq(model, tokenizer, items, device, "biology", batch_size=4)
    assert result["n"] == 8
    assert 0.0 <= result["accuracy"] <= 100.0


def test_generation_produces_one_continuation_per_prefix(loaded):
    cfg, model, tokenizer, device, _ = loaded
    prefixes = ["Antiviral medications work by blocking", "Vaccines protect populations by"]
    gens = generate_continuations(model, tokenizer, prefixes, device, max_new_tokens=5)
    assert [g.prefix for g in gens] == prefixes
    assert len(gens) == 2


def test_full_train_loop_reduces_target_feature_activation(loaded):
    cfg, model, tokenizer, device, dtype = loaded
    cfg.train.steps = 12
    cfg.train.lr = 5e-3
    cfg.train.beta = 0.0  # isolate the unlearning term for this assertion
    cfg.train.gamma = 0.0
    cfg.train.alpha = 1.0

    saes = load_saes(cfg.sae, cfg.model.sae_layers, model.config.hidden_size, device, dtype)
    target = ["viral replication in host cells requires assembly of new virions"] * 8
    retain = ["balanced trees provide predictable lookup costs for collections"] * 8
    features = run_selection(model, tokenizer, saes, target, retain, cfg, device)
    assert features.total > 0

    peft_model = attach_lora(model, cfg)
    result = train_crisp(
        peft_model, tokenizer, saes, features, target, retain, retain, cfg, device
    )
    history = result["history"]
    assert len(history) == cfg.train.steps
    assert history[-1]["unlearn"] < history[0]["unlearn"]
    peft_model.unload()


def test_rmu_baseline_runs_and_updates_weights(loaded):
    from crisp.baselines.rmu import RMUConfig, train_rmu

    cfg, model, tokenizer, device, _ = loaded
    rmu_cfg = RMUConfig(layer_ids=[0, 1], steps=3, batch_size=2, lr=1e-3, max_seq_len=32)
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if "layers.1.mlp.down_proj" in n
    }
    history = train_rmu(
        model, tokenizer, ["viral replication in host cells"] * 4,
        ["balanced trees have predictable lookup costs"] * 4, rmu_cfg, device,
    )["history"]
    assert len(history) == 3
    assert all(h["forget"] > 0 for h in history)
    assert any(
        not torch.allclose(before[n], p)
        for n, p in model.named_parameters()
        if n in before
    )
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in before:
                p.copy_(before[n])
        model.requires_grad_(True)


def test_elm_baseline_runs(loaded):
    from crisp.baselines.elm import ELMConfig, train_elm

    cfg, model, tokenizer, device, _ = loaded
    peft_model = attach_lora(model, cfg)
    elm_cfg = ELMConfig(steps=2, batch_size=2, max_seq_len=32, eta=10.0, domain="bio")
    history = train_elm(
        peft_model, tokenizer, ["viral replication in host cells"] * 4,
        ["balanced trees have predictable lookup costs"] * 4, elm_cfg, device,
    )["history"]
    assert len(history) == 2
    assert all(h["loss"] == h["loss"] for h in history)  # not NaN
    peft_model.unload()


def test_coherence_uses_the_final_layer(loaded):
    _, model, *_ = loaded
    assert len(decoder_layers(model)) == model.config.num_hidden_layers
