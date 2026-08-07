"""Corpora, MCQ benchmarks and coherence sets (Section 4.1)."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .utils import REPO_ROOT, get_logger, hf_token

log = get_logger(__name__)

#: Everything the pipeline reads lives under ``data/``. ``crisp fetch``
#: materialises the Hugging Face corpora and MCQ benchmarks here once, after
#: which every run is local (and therefore offline-reproducible).
DATA_ROOT = REPO_ROOT / "data"
CORPUS_DIR = DATA_ROOT / "wmdp"
MCQ_DIR = DATA_ROOT / "mcq"

# Most WMDP corpora ship as parquet configs of a single HF repo. The bio forget
# corpus is not published there: it is gated and lives in its own repo with a
# single default config.
WMDP_CORPORA_REPO = "cais/wmdp-corpora"
WMDP_BIO_FORGET_REPO = "cais/wmdp-bio-forget-corpus"

#: ``(repo_id, config_name)`` per corpus. ``config_name`` is ``None`` for repos
#: that expose a single default config.
WMDP_CORPUS_SOURCES: dict[str, tuple[str, str | None]] = {
    "bio_target": (WMDP_BIO_FORGET_REPO, None),
    "bio_retain": (WMDP_CORPORA_REPO, "bio-retain-corpus"),
    "cyber_target": (WMDP_CORPORA_REPO, "cyber-forget-corpus"),
    "cyber_retain": (WMDP_CORPORA_REPO, "cyber-retain-corpus"),
}
#: Raw jsonl filenames used by older uploads/mirrors, tried if the config fails.
WMDP_CORPUS_FILES: dict[str, list[str]] = {
    "bio_target": ["bio_remove_dataset.jsonl", "bio-forget-corpus.jsonl"],
    "bio_retain": ["bio-retain-corpus.jsonl", "bio_retain_dataset.jsonl"],
    "cyber_target": ["cyber-forget-corpus.jsonl", "cyber_remove_dataset.jsonl"],
    "cyber_retain": ["cyber-retain-corpus.jsonl", "cyber_retain_dataset.jsonl"],
}

WMDP_MCQ_CONFIG = {"bio": "wmdp-bio", "cyber": "wmdp-cyber"}

# In-domain retention MCQs (Section 4.1).
MMLU_SUBJECTS = {
    "bio": ["high_school_biology", "college_biology"],
    "cyber": ["high_school_computer_science", "college_computer_science"],
}

CONCEPT_NAME = {"bio": "biosecurity", "cyber": "cybersecurity"}

_MARKDOWN_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", flags=re.MULTILINE)
_IMAGE_LINK = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_CITATION = re.compile(r"\[(?:\d+(?:\s*[-,–]\s*\d+)*)\]|\((?:[A-Z][A-Za-z.'-]+"
                       r"(?:\s+et\s+al\.)?(?:\s*,\s*)?\d{4}[a-z]?)\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE = re.compile(r"[ \t]{2,}")


def clean_document(text: str, max_chars: int = 1000) -> str:
    """Remove markdown headers, citations, image links and non-ASCII characters,
    then right-truncate to ``max_chars`` (Section 4.1)."""
    text = _IMAGE_LINK.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _MARKDOWN_HEADER.sub("", text)
    text = _CITATION.sub(" ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def _extract_text(row: dict) -> str:
    for key in ("text", "abstract", "content", "document", "passage"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # Fall back to the longest string field.
    strings = [v for v in row.values() if isinstance(v, str)]
    return max(strings, key=len) if strings else ""


def _read_jsonl(path: str | Path) -> list[str]:
    texts: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(_extract_text(json.loads(line)))
            except json.JSONDecodeError:
                texts.append(line)
    return texts


def _load_local(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
        return [_extract_text(r) if isinstance(r, dict) else str(r) for r in rows]
    return [b for b in path.read_text(errors="ignore").split("\n\n") if b.strip()]


def local_corpus_path(domain: str, role: str) -> Path:
    """Where ``crisp fetch`` stores the raw corpus for ``<domain>/<role>``."""
    return CORPUS_DIR / f"{domain}_{role}.jsonl"


def local_mcq_path(name: str) -> Path:
    """Where ``crisp fetch`` stores a materialised MCQ benchmark."""
    return MCQ_DIR / f"{name}.jsonl"


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _read_mcq_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _download_wmdp_corpus(key: str, repo_override: str | None = None) -> list[str]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    default_repo, config_name = WMDP_CORPUS_SOURCES[key]
    repo_id = repo_override or default_repo
    if repo_override and repo_override != default_repo:
        config_name = None  # a custom repo is assumed to expose a default config
    errors = []

    # Preferred path: the published parquet configs.
    try:
        from datasets import load_dataset

        kwargs = {"name": config_name} if config_name else {}
        ds = load_dataset(repo_id, split="train", token=hf_token(), **kwargs)
        return [_extract_text(row) for row in ds]
    except Exception as exc:  # no access (gated), config missing, or offline
        errors.append(f"config {config_name or 'default'}: {type(exc).__name__}")

    # Fallback: raw jsonl uploads used by some mirrors of the forget corpora.
    for filename in WMDP_CORPUS_FILES[key]:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=hf_token(),
            )
            return _read_jsonl(path)
        except (HfHubHTTPError, OSError, ValueError) as exc:  # not found / gated
            errors.append(f"{filename}: {type(exc).__name__}")
    raise FileNotFoundError(
        f"could not fetch WMDP corpus {key!r} from {repo_id} ({'; '.join(errors)}).\n"
        f"If it is gated, request access at https://huggingface.co/datasets/{repo_id}, "
        "run `hf auth login`, or point data.target_corpus / data.retain_corpus at a "
        "local .jsonl file."
    )


def load_corpus(
    domain: str,
    role: str,
    override_path: str | None,
    max_docs: int,
    max_chars: int,
    seed: int,
    repo_override: str | None = None,
) -> list[str]:
    """Load and preprocess one corpus. ``role`` is ``target`` or ``retain``."""
    key = f"{domain}_{role}"
    cached = local_corpus_path(domain, role)
    if override_path:
        raw = _load_local(override_path)
        source = override_path
    elif cached.is_file() and not repo_override:
        raw = _load_local(cached)
        source = str(cached)
    else:
        raw = _download_wmdp_corpus(key, repo_override)
        repo, config_name = WMDP_CORPUS_SOURCES[key]
        source = repo_override or (f"{repo}:{config_name}" if config_name else repo)

    docs = [clean_document(t, max_chars) for t in raw]
    docs = [d for d in docs if len(d) >= 50]
    if max_docs and len(docs) > max_docs:
        # WMDP-Bio: sample 5000 entries at random; WMDP-Cyber uses all 986.
        docs = random.Random(seed).sample(docs, max_docs)
    log.info("loaded %d docs for %s/%s from %s", len(docs), domain, role, source)
    return docs


@dataclass
class MCQItem:
    question: str
    choices: list[str]
    answer: int

    def prompt(self, header: str = "") -> str:
        letters = "ABCD"
        lines = [header] if header else []
        lines.append(f"{self.question.strip()}")
        for letter, choice in zip(letters, self.choices):
            lines.append(f"{letter}. {choice}")
        lines.append("Answer:")
        return "\n".join(lines)


def _to_items(rows) -> list[MCQItem]:
    items = []
    for row in rows:
        choices = list(row["choices"])
        if len(choices) != 4:
            continue
        items.append(MCQItem(row["question"], choices, int(row["answer"])))
    return items


def split_half(items: list[MCQItem], split: str, seed: int = 0) -> list[MCQItem]:
    """Divide MCQs evenly into validation and test splits (Section 4.1)."""
    if split == "all":
        return items
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    mid = len(order) // 2
    keep = set(order[:mid]) if split == "validation" else set(order[mid:])
    return [it for i, it in enumerate(items) if i in keep]


def _mcq_rows(name: str, repo: str, config: str) -> list[dict]:
    """Rows for one MCQ benchmark, from ``data/mcq`` if fetched, else the Hub."""
    cached = local_mcq_path(name)
    if cached.is_file():
        return _read_mcq_jsonl(cached)

    from datasets import load_dataset

    ds = load_dataset(repo, config, split="test", token=hf_token())
    log.info("loaded MCQ set %s from %s:%s (run `crisp fetch` to cache it under "
             "data/mcq)", name, repo, config)
    return [dict(row) for row in ds]


def load_wmdp_mcq(domain: str, split: str = "test", seed: int = 0) -> list[MCQItem]:
    rows = _mcq_rows(f"wmdp-{domain}", "cais/wmdp", WMDP_MCQ_CONFIG[domain])
    return split_half(_to_items(rows), split, seed)


def load_mmlu(
    subjects: list[str], split: str = "test", seed: int = 0, max_per_subject: int = 0
) -> list[MCQItem]:
    items: list[MCQItem] = []
    for subject in subjects:
        rows = _mcq_rows(f"mmlu-{subject}", "cais/mmlu", subject)
        subject_items = _to_items(rows)
        if max_per_subject:
            subject_items = subject_items[:max_per_subject]
        items.extend(split_half(subject_items, split, seed))
    return items


def load_mmlu_full(max_per_subject: int = 10, split: str = "all") -> list[MCQItem]:
    """Full-MMLU utility metric. The paper's hyperparameter selection uses the
    first 10 questions of each subject."""
    rows = _mcq_rows("mmlu-all", "cais/mmlu", "all")
    by_subject: dict[str, list[MCQItem]] = {}
    for row in rows:
        subject = row.get("subject", "all")
        bucket = by_subject.setdefault(subject, [])
        if max_per_subject and len(bucket) >= max_per_subject:
            continue
        bucket.append(MCQItem(row["question"], list(row["choices"]), int(row["answer"])))
    items = [it for bucket in by_subject.values() for it in bucket]
    return items


def load_coherence_set(domain: str, path: str | None = None) -> list[str]:
    """20 benign, factual sentences per domain (Appendix D)."""
    if path is None:
        path = DATA_ROOT / "coherence" / f"{domain}.json"
    return json.loads(Path(path).read_text())


def load_gen_prompts(domain: str, path: str | None = None) -> list[str]:
    """100 natural-language prefixes for fluency/concept scoring (Appendix E)."""
    if path is None:
        path = DATA_ROOT / "prompts" / f"{domain}.json"
    return json.loads(Path(path).read_text())
