"""Unit tests for the CRISP equations and data handling."""

from __future__ import annotations

import json

import pytest
import torch

from crisp.config import Config
from crisp.data import MCQItem, clean_document, split_half
from crisp.features import LayerStats, select_features
from crisp.losses import representation_distance, total_loss, unlearning_loss
from crisp.metrics import overall_score, selection_score
from crisp.sae import SAEBundle, SparseAutoencoder
from crisp.utils import harmonic_mean


# --- Section 3.2: feature selection -------------------------------------------


def _stats(count, total, n_tokens):
    return LayerStats(
        count=torch.tensor(count, dtype=torch.float64),
        total=torch.tensor(total, dtype=torch.float64),
        n_tokens=n_tokens,
    )


def test_select_features_applies_topk_then_tau():
    # feature 0: frequent on target and strongly target-skewed  -> selected
    # feature 1: frequent on target but ratio below tau         -> filtered out
    # feature 2: rarely active on target                        -> not in top-k
    target = _stats([100.0, 90.0, 1.0], [1000.0, 100.0, 5.0], 100)
    retain = _stats([1.0, 50.0, 1.0], [10.0, 90.0, 5.0], 100)
    sel = select_features(
        {0: target}, {0: retain}, top_k=2, tau=3.0, epsilon=1e-6
    )
    assert sel.layers[0] == [0]
    assert sel.scores[0]["feature"][:2] == [0, 1]


def test_select_features_normalises_unequal_corpus_sizes():
    # Same per-token rate on both corpora: nothing should be salient.
    target = _stats([200.0], [2000.0], 200)
    retain = _stats([100.0], [1000.0], 100)
    sel = select_features({0: target}, {0: retain}, top_k=1, tau=3.0)
    assert sel.layers[0] == []
    assert sel.scores[0]["delta_phi"][0] == pytest.approx(0.0)
    assert sel.scores[0]["rho"][0] == pytest.approx(1.0, abs=1e-6)


def test_selected_features_roundtrip(tmp_path):
    from crisp.features import SelectedFeatures

    sel = SelectedFeatures(layers={4: [1, 2], 6: []}, scores={}, top_k=5, tau=3.0)
    path = tmp_path / "features.json"
    sel.to_json(path)
    loaded = SelectedFeatures.from_json(path)
    assert loaded.layers == {4: [1, 2], 6: []}
    assert loaded.total == 2


# --- Section 3.1: SAE ---------------------------------------------------------


def _toy_sae(threshold=0.5):
    W_enc = torch.eye(3)
    return SparseAutoencoder(
        W_enc=W_enc,
        b_enc=torch.zeros(3),
        W_dec=W_enc.clone(),
        b_dec=torch.zeros(3),
        activation="jumprelu",
        threshold=torch.full((3,), threshold),
    )


def test_jumprelu_zeroes_below_threshold():
    sae = _toy_sae(threshold=0.5)
    acts = sae.encode(torch.tensor([[0.2, 0.7, -1.0]]))
    assert torch.allclose(acts, torch.tensor([[0.0, 0.7, 0.0]]))


def test_relu_and_topk_activations():
    W = torch.eye(4)
    relu_sae = SparseAutoencoder(W, torch.zeros(4), W.clone(), torch.zeros(4), "relu")
    assert torch.allclose(
        relu_sae.encode(torch.tensor([[1.0, -1.0, 2.0, 0.0]])),
        torch.tensor([[1.0, 0.0, 2.0, 0.0]]),
    )
    topk_sae = SparseAutoencoder(W, torch.zeros(4), W.clone(), torch.zeros(4), "topk", k=2)
    assert torch.allclose(
        topk_sae.encode(torch.tensor([[1.0, -1.0, 3.0, 0.5]])),
        torch.tensor([[1.0, 0.0, 3.0, 0.0]]),
    )


# --- Section 3.3: losses ------------------------------------------------------


def test_unlearning_loss_matches_equation_9():
    sae = _toy_sae(threshold=0.0)
    bundle = SAEBundle({0: sae})
    hidden = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]]])  # [1, 2, 3]
    mask = torch.ones(1, 2, dtype=torch.bool)
    lam = 2.0
    loss = unlearning_loss({0: hidden}, mask, bundle, {0: [0]}, lam)

    # token 0: a_0 = 1, c = (1+2+3)/3 = 2  -> 1 + 2*2 = 5
    # token 1: a_0 = 4, c = 4/3            -> 4 + 2*4/3 = 6.6667
    expected = (5.0 + (4.0 + 2.0 * 4.0 / 3.0)) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_unlearning_loss_requires_salient_features():
    bundle = SAEBundle({0: _toy_sae()})
    with pytest.raises(ValueError):
        unlearning_loss(
            {0: torch.zeros(1, 2, 3)}, torch.ones(1, 2, dtype=torch.bool), bundle, {0: []}, 1.0
        )


def test_unlearning_loss_is_differentiable_wrt_hidden_states():
    sae = _toy_sae(threshold=0.0)
    bundle = SAEBundle({0: sae})
    hidden = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    loss = unlearning_loss(
        {0: hidden}, torch.ones(1, 1, dtype=torch.bool), bundle, {0: [0]}, 0.0
    )
    loss.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_representation_distance_reductions():
    a = {0: torch.tensor([[[1.0, 1.0]]])}
    b = {0: torch.tensor([[[0.0, 0.0]]])}
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert representation_distance(a, b, mask, "sqnorm").item() == pytest.approx(2.0)
    assert representation_distance(a, b, mask, "mse").item() == pytest.approx(1.0)
    assert representation_distance(a, a, mask).item() == pytest.approx(0.0)


def test_total_loss_weights_eq_11():
    out = total_loss(
        torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0), alpha=0.01, beta=0.99, gamma=0.01
    )
    assert out.item() == pytest.approx(0.01 * 2 + 0.99 * 3 + 0.01 * 4)


# --- Section 4.4: metrics -----------------------------------------------------


def test_overall_score_is_harmonic_mean_eq_12():
    value = overall_score(unlearn_acc=30.0, retain_acc=74.0, mmlu_acc=60.0, fluency=0.8, concept=1.6)
    assert value == pytest.approx(harmonic_mean([70.0, 74.0, 60.0, 40.0, 80.0]))


def test_overall_score_penalises_a_collapsed_axis():
    good = overall_score(30.0, 74.0, 60.0, 0.8, 1.6)
    degenerate = overall_score(30.0, 74.0, 60.0, 0.0, 1.6)
    assert degenerate == 0.0 < good


def test_selection_score_is_one_for_an_unchanged_model():
    assert selection_score(60.0, 60.0, 70.0, 70.0, 50.0, 50.0) == pytest.approx(1.0)


def test_selection_score_rewards_forgetting():
    forgot = selection_score(30.0, 60.0, 70.0, 70.0, 50.0, 50.0)
    assert forgot > 1.0


# --- Section 4.1: data --------------------------------------------------------


def test_clean_document_strips_artifacts_and_truncates():
    raw = (
        "## Header\n"
        "Some text [12] with a citation (Smith et al., 2024) and an ![img](http://x/y.png) "
        "image plus a café non-ascii word and https://example.com/link.\n"
    )
    cleaned = clean_document(raw, max_chars=1000)
    assert "##" not in cleaned
    assert "[12]" not in cleaned
    assert "Smith et al." not in cleaned
    assert "http" not in cleaned
    assert cleaned.isascii()
    assert len(clean_document("x" * 5000, max_chars=1000)) == 1000


def test_split_half_partitions_mcqs_without_overlap():
    items = [MCQItem(f"q{i}", ["a", "b", "c", "d"], 0) for i in range(11)]
    val = split_half(items, "validation", seed=0)
    test = split_half(items, "test", seed=0)
    assert len(val) + len(test) == len(items)
    assert not {q.question for q in val} & {q.question for q in test}
    assert split_half(items, "all") == items


def test_mcq_prompt_format():
    item = MCQItem("What is 2+2?", ["3", "4", "5", "6"], 1)
    prompt = item.prompt("Header\n")
    assert prompt.endswith("Answer:")
    assert "A. 3" in prompt and "D. 6" in prompt


# --- config -------------------------------------------------------------------


def test_config_defaults_follow_appendix_f():
    cfg = Config()
    assert cfg.train.beta == 0.99
    assert cfg.train.gamma == 0.01
    assert cfg.train.resolved_alpha == pytest.approx(0.01)
    assert cfg.train.resolved_lora_alpha == 16
    assert cfg.selection.tau == 3.0


def test_config_overrides_and_unknown_keys(tmp_path):
    cfg = Config().apply_overrides(["train.lr=0.001", "selection.top_k=50"])
    assert cfg.train.lr == 0.001 and cfg.selection.top_k == 50
    with pytest.raises(ValueError):
        Config().apply_overrides(["train.nonexistent=1"])

    path = tmp_path / "c.yaml"
    path.write_text("data:\n  domain: cyber\n")
    assert Config.from_yaml(path).data.domain == "cyber"

    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  nonsense: 1\n")
    with pytest.raises(ValueError):
        Config.from_yaml(bad)


def test_bio_forget_corpus_comes_from_its_own_repo():
    from crisp.data import WMDP_BIO_FORGET_REPO, WMDP_CORPORA_REPO, WMDP_CORPUS_SOURCES

    assert WMDP_CORPUS_SOURCES["bio_target"] == (WMDP_BIO_FORGET_REPO, None)
    for key in ("bio_retain", "cyber_target", "cyber_retain"):
        repo, config_name = WMDP_CORPUS_SOURCES[key]
        assert repo == WMDP_CORPORA_REPO and config_name


def test_bio_configs_leave_corpus_resolution_to_the_data_dir():
    """Bio configs must not pin a corpus repo: an override disables the
    data/wmdp cache that `crisp fetch` fills, forcing a Hub download per run."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    paths = sorted(root.glob("*_bio*.yaml"))
    assert paths
    for path in paths:
        cfg = Config.from_yaml(path)
        assert cfg.data.target_corpus_repo is None, path.name
        assert cfg.data.retain_corpus_repo is None, path.name


def test_shipped_assets_have_expected_sizes():
    from crisp.data import load_coherence_set, load_gen_prompts

    for domain in ("bio", "cyber"):
        assert len(load_coherence_set(domain)) == 20
        assert len(load_gen_prompts(domain)) == 100


def test_all_paper_configs_parse():
    from pathlib import Path

    for path in sorted(Path("configs").glob("*.yaml")):
        cfg = Config.from_yaml(path)
        assert cfg.model.sae_layers
        assert json.dumps(cfg.to_dict())


def test_rating_parser_ignores_the_reasoning_block():
    from crisp.eval_gen import _parse_rating

    reply = (
        "<think>The text repeats itself, so maybe 0, or possibly 1 or 2.</think>\n"
        "The generated text is repetitive. Rating: [[1]]"
    )
    assert _parse_rating(reply) == 1

    # No closing tag -> the rater was truncated mid-reasoning and never rated.
    assert _parse_rating("<think>Hmm, this looks like a 2 to me and") is None

    # A non-thinking rater still parses, including via the trailing-digit fallback.
    assert _parse_rating("Fluent enough. Rating: [[2]]") == 2
    assert _parse_rating("The concept is absent, so the score is 0") == 0
