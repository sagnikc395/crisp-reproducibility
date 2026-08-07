"""Local dataset resolution (data/) and result reporting (artifacts/results)."""

from __future__ import annotations

import json

import pytest

from crisp import data, report
from crisp.utils import load_dotenv


def _write_corpus(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"text": t}) for t in texts))


def test_load_corpus_prefers_the_fetched_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CORPUS_DIR", tmp_path / "wmdp")
    _write_corpus(data.local_corpus_path("bio", "target"), ["a fetched document " * 10])

    def fail(*args, **kwargs):
        raise AssertionError("hit the Hub despite a local corpus being present")

    monkeypatch.setattr(data, "_download_wmdp_corpus", fail)
    docs = data.load_corpus("bio", "target", None, 0, 1000, 0)
    assert len(docs) == 1 and docs[0].startswith("a fetched document")


def test_repo_override_still_bypasses_the_local_cache(tmp_path, monkeypatch):
    """An explicit data.*_corpus_repo must win, so a user can point a run at a
    mirror without deleting data/wmdp first."""
    monkeypatch.setattr(data, "CORPUS_DIR", tmp_path / "wmdp")
    _write_corpus(data.local_corpus_path("bio", "target"), ["local " * 30])
    monkeypatch.setattr(data, "_download_wmdp_corpus", lambda key, repo=None: ["remote " * 30])

    docs = data.load_corpus("bio", "target", None, 0, 1000, 0, "someone/mirror")
    assert docs[0].startswith("remote")


def test_local_corpus_is_ignored_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CORPUS_DIR", tmp_path / "empty")
    monkeypatch.setattr(data, "_download_wmdp_corpus", lambda key, repo=None: ["remote " * 30])
    assert data.load_corpus("bio", "target", None, 0, 1000, 0)[0].startswith("remote")


def test_mcq_rows_read_the_fetched_file(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MCQ_DIR", tmp_path / "mcq")
    rows = [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4}
            for i in range(8)]
    data.write_jsonl(data.local_mcq_path("wmdp-bio"), rows)

    items = data.load_wmdp_mcq("bio", split="all")
    assert len(items) == 8
    assert items[0].question == "q0" and items[0].answer == 0
    # The halves partition the set exactly.
    val = data.load_wmdp_mcq("bio", split="validation")
    test = data.load_wmdp_mcq("bio", split="test")
    assert len(val) + len(test) == 8
    assert {i.question for i in val}.isdisjoint({i.question for i in test})


def test_fetch_manifest_covers_everything_a_domain_needs(tmp_path, monkeypatch):
    from crisp import fetch

    monkeypatch.setattr(data, "CORPUS_DIR", tmp_path / "wmdp")
    monkeypatch.setattr(data, "MCQ_DIR", tmp_path / "mcq")
    assert fetch.missing_datasets("bio")  # nothing fetched yet

    for role in ("target", "retain"):
        _write_corpus(data.local_corpus_path("bio", role), ["doc " * 30])
    for name in ["wmdp-bio", "mmlu-all", *(f"mmlu-{s}" for s in data.MMLU_SUBJECTS["bio"])]:
        data.write_jsonl(data.local_mcq_path(name), [{"question": "q"}])
    assert fetch.missing_datasets("bio") == []


def test_rmu_layer_window_fits_small_models():
    from crisp.baselines.rmu import resolve_layer_ids

    assert resolve_layer_ids([5, 6, 7], 26) == [5, 6, 7]  # paper default, untouched
    assert resolve_layer_ids([5, 6, 7], 2) == [0, 1]  # smoke model
    assert resolve_layer_ids([5, 6, 7], 8) == [5, 6, 7]
    assert resolve_layer_ids([5, 6, 7], 6) == [3, 4, 5]  # window slides down to fit


def test_corpus_statistics_accumulate_off_device():
    """float64 accumulators must stay on the CPU: MPS has no float64 at all."""
    import torch

    from crisp.features import corpus_statistics

    class _SAE:
        d_sae = 6

        def encode(self, hidden):
            return torch.arange(6, dtype=torch.float32).repeat(hidden.shape[0], 1)

    class _Bundle:
        layers = [0]

        def __getitem__(self, _):
            return _SAE()

    class _Model:
        def __call__(self, **kwargs):
            return None

    import crisp.features as features

    class _Capture:
        def __init__(self, *a, **k):
            self.acts = {0: torch.ones(1, 4, 3)}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    features_capture = features.ResidualCapture
    features.ResidualCapture = _Capture
    tokenize = features.tokenize_batch
    features.tokenize_batch = lambda tok, docs, n, dev: {
        "attention_mask": torch.ones(1, 4, dtype=torch.long)
    }
    try:
        stats = corpus_statistics(
            _Model(), None, _Bundle(), ["doc"], 8, 1, torch.device("cpu")
        )
    finally:
        features.ResidualCapture = features_capture
        features.tokenize_batch = tokenize

    layer = stats[0]
    assert layer.count.dtype == torch.float64 and layer.count.device.type == "cpu"
    assert layer.total.dtype == torch.float64 and layer.total.device.type == "cpu"
    assert layer.n_tokens == 4
    # feature i activates on every one of the 4 tokens with magnitude i
    assert layer.total.tolist() == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    assert layer.count.tolist() == [0.0, 4.0, 4.0, 4.0, 4.0, 4.0]


def _result(**kwargs):
    base = {"domain": "bio", "split": "test", "unlearn_acc": 30.0,
            "retain_acc": 60.0, "mmlu": 50.0}
    base.update(kwargs)
    return base


def test_report_orders_methods_and_recomputes_overall(tmp_path):
    for run, payload in [
        ("demo_elm", _result(fluency=2.0, concept=0.5)),
        ("demo_original", _result(unlearn_acc=70.0)),
        ("demo_crisp", _result(unlearn_acc=25.0)),
    ]:
        (tmp_path / f"{run}__test.json").write_text(json.dumps(payload))
    (tmp_path / "summary.json").write_text(json.dumps({"n_runs": 0}))  # must be ignored

    summary = report.build(tmp_path)
    assert [r["method"] for r in summary["runs"]] == ["original", "crisp", "elm"]
    assert all(r["config"] == "demo" for r in summary["runs"])

    by_method = {r["method"]: r for r in summary["runs"]}
    assert by_method["crisp"]["overall"] is None  # unjudged -> no overall
    assert by_method["elm"]["overall"] == pytest.approx(
        report.overall_score(30.0, 60.0, 50.0, 2.0, 0.5)
    )

    table = (tmp_path / "README.md").read_text()
    assert "demo_crisp" in table and "| - |" in table


def test_result_path_is_under_artifacts_results():
    path = report.result_path("gemma2-2b_bio_crisp", "test")
    assert path.parent == report.RESULTS_DIR
    assert path.parent.parent.name == "artifacts"
    assert path.name == "gemma2-2b_bio_crisp__test.json"


def test_dotenv_does_not_clobber_the_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('export ALREADY_SET="from-file"\n# comment\nFRESH_KEY=abc123\n\n')
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    monkeypatch.delenv("FRESH_KEY", raising=False)

    load_dotenv(env)
    import os

    assert os.environ["ALREADY_SET"] == "from-shell"
    assert os.environ["FRESH_KEY"] == "abc123"
