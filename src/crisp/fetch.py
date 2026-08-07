"""Materialise every dataset the pipeline needs under ``data/``.

Selection, training and evaluation all read from ``data/`` first, so running
``crisp fetch`` once pins the exact corpora and benchmarks a reproduction used
and makes later runs independent of the Hub (and of gated-access changes).

Layout produced::

    data/wmdp/{bio,cyber}_{target,retain}.jsonl   raw corpora, one {"text": ...} per line
    data/mcq/wmdp-{bio,cyber}.jsonl              WMDP multiple-choice benchmarks
    data/mcq/mmlu-{subject}.jsonl                in-domain MMLU retention subjects
    data/mcq/mmlu-all.jsonl                      full MMLU utility benchmark
    data/MANIFEST.json                           source repo + row count per file
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import (
    DATA_ROOT,
    MMLU_SUBJECTS,
    WMDP_CORPUS_SOURCES,
    WMDP_MCQ_CONFIG,
    _download_wmdp_corpus,
    local_corpus_path,
    local_mcq_path,
    write_jsonl,
)
from .report import _display_path
from .utils import get_logger, hf_token

log = get_logger(__name__)

MANIFEST_PATH = DATA_ROOT / "MANIFEST.json"


def _fetch_corpus(domain: str, role: str, force: bool) -> dict:
    path = local_corpus_path(domain, role)
    repo, config = WMDP_CORPUS_SOURCES[f"{domain}_{role}"]
    source = f"{repo}:{config}" if config else repo
    if path.is_file() and not force:
        log.info("corpus %s already present, skipping", path)
        return {"path": _display_path(path), "source": source,
                "rows": sum(1 for _ in open(path, encoding="utf-8")), "cached": True}

    texts = _download_wmdp_corpus(f"{domain}_{role}")
    # Store raw text: cleaning and truncation stay in load_corpus so that
    # data.max_chars remains a config knob rather than being baked in here.
    write_jsonl(path, [{"text": t} for t in texts if t and t.strip()])
    log.info("wrote %d docs to %s", len(texts), path)
    return {"path": _display_path(path), "source": source,
            "rows": len(texts), "cached": False}


def _fetch_mcq(name: str, repo: str, config: str, force: bool) -> dict:
    path = local_mcq_path(name)
    source = f"{repo}:{config}"
    if path.is_file() and not force:
        log.info("MCQ set %s already present, skipping", path)
        return {"path": _display_path(path), "source": source,
                "rows": sum(1 for _ in open(path, encoding="utf-8")), "cached": True}

    from datasets import load_dataset

    ds = load_dataset(repo, config, split="test", token=hf_token())
    keep = ("question", "choices", "answer", "subject")
    rows = [{k: row[k] for k in keep if k in row} for row in ds]
    write_jsonl(path, rows)
    log.info("wrote %d questions to %s", len(rows), path)
    return {"path": _display_path(path), "source": source,
            "rows": len(rows), "cached": False}


def fetch_all(
    domains: list[str] | None = None,
    force: bool = False,
    skip_corpora: bool = False,
    skip_mcq: bool = False,
) -> dict:
    """Download everything for ``domains`` (default: bio and cyber).

    Returns only the entries this call covered; the manifest on disk accumulates
    across calls so a bio run does not drop what a previous cyber run fetched.
    """
    domains = domains or ["bio", "cyber"]
    manifest: dict = {}
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text())
    before = set(manifest)
    touched: list[str] = []

    for domain in domains:
        if not skip_corpora:
            for role in ("target", "retain"):
                key = f"corpus/{domain}_{role}"
                manifest[key] = _fetch_corpus(domain, role, force)
                touched.append(key)
        if not skip_mcq:
            key = f"mcq/wmdp-{domain}"
            manifest[key] = _fetch_mcq(
                f"wmdp-{domain}", "cais/wmdp", WMDP_MCQ_CONFIG[domain], force
            )
            touched.append(key)
            for subject in MMLU_SUBJECTS[domain]:
                key = f"mcq/mmlu-{subject}"
                manifest[key] = _fetch_mcq(f"mmlu-{subject}", "cais/mmlu", subject, force)
                touched.append(key)

    if not skip_mcq:
        manifest["mcq/mmlu-all"] = _fetch_mcq("mmlu-all", "cais/mmlu", "all", force)
        touched.append("mcq/mmlu-all")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    log.info("manifest at %s covers %d file(s), %d new this run",
             MANIFEST_PATH, len(manifest), len(set(touched) - before))
    return {k: manifest[k] for k in touched}


def missing_datasets(domain: str) -> list[Path]:
    """Local files ``domain`` needs that ``crisp fetch`` has not produced yet."""
    wanted = [local_corpus_path(domain, "target"), local_corpus_path(domain, "retain"),
              local_mcq_path(f"wmdp-{domain}"), local_mcq_path("mmlu-all")]
    wanted += [local_mcq_path(f"mmlu-{s}") for s in MMLU_SUBJECTS[domain]]
    return [p for p in wanted if not p.is_file()]
