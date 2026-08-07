"""Result files and the Table 1 comparison, all under ``artifacts/results``."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import overall_score  # re-exported: tests and notebooks score rows
from .utils import REPO_ROOT, get_logger

log = get_logger(__name__)

RESULTS_DIR = REPO_ROOT / "artifacts" / "results"


def _display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (custom --results-dir)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

#: Column order of Table 1. ``unlearn_acc`` is lower-is-better; the rest are
#: higher-is-better.
COLUMNS = [
    ("unlearn_acc", "WMDP acc", "down"),
    ("retain_acc", "In-domain MMLU", "up"),
    ("mmlu", "MMLU", "up"),
    ("fluency", "Fluency", "up"),
    ("concept", "Concept", "down"),
    ("overall", "Overall", "up"),
]

#: Suffixes ``reproduce.sh`` appends to a config name, in reporting order.
METHOD_ORDER = ["original", "crisp", "rmu", "elm"]


def result_path(run_name: str, split: str = "test") -> Path:
    """Canonical location of one evaluation's results."""
    return RESULTS_DIR / f"{run_name}__{split}.json"


def _method_of(run_name: str) -> str:
    for method in METHOD_ORDER:
        if run_name.endswith(f"_{method}"):
            return method
    return run_name


def _sort_key(entry: dict) -> tuple:
    method = entry["method"]
    rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (entry["config"], rank, entry["run"])


def collect(results_dir: Path | str = RESULTS_DIR) -> list[dict]:
    """Load every result JSON in ``results_dir`` into flat table rows."""
    results_dir = Path(results_dir)
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*__*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("skipping unreadable result file %s", path)
            continue
        if not isinstance(data, dict) or "unlearn_acc" not in data:
            continue  # aggregate outputs (summary.json) and other stray files
        run = path.stem.rsplit("__", 1)[0]
        method = _method_of(run)
        entry = {
            "run": run,
            "method": method,
            "config": run[: -len(method) - 1] if run.endswith(f"_{method}") else run,
            "domain": data.get("domain"),
            "split": data.get("split"),
            "source": _display_path(path),
        }
        for key, _, _ in COLUMNS:
            value = data.get(key)
            entry[key] = float(value) if isinstance(value, (int, float)) else None
        # Overall is only defined once the judged columns exist; recompute so a
        # run evaluated with --no-judge and one judged later stay consistent.
        if entry["overall"] is None and all(
            entry[k] is not None for k in ("unlearn_acc", "retain_acc", "mmlu", "fluency", "concept")
        ):
            entry["overall"] = overall_score(
                entry["unlearn_acc"], entry["retain_acc"], entry["mmlu"],
                entry["fluency"], entry["concept"],
            )
        rows.append(entry)
    return sorted(rows, key=_sort_key)


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "No results yet. Run `scripts/reproduce.sh <config.yaml>` first.\n"

    headers = ["Run", "Method"] + [f"{label} {'v' if d == 'down' else '^'}"
                                   for _, label, d in COLUMNS]
    lines = [
        "# CRISP reproduction results",
        "",
        "`^` higher is better, `v` lower is better. Blank cells were not evaluated",
        "(generation columns are scored by the local Qwen3 rater; `--no-judge` "
        "leaves them empty).",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = [row["run"], row["method"]]
        cells += [f"{row[key]:.2f}" if row[key] is not None else "-" for key, _, _ in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def build(results_dir: Path | str = RESULTS_DIR) -> dict:
    """Aggregate the result JSONs into ``summary.json`` and ``README.md``."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(results_dir)
    summary = {"n_runs": len(rows), "columns": [c[0] for c in COLUMNS], "runs": rows}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "README.md").write_text(to_markdown(rows))
    log.info("aggregated %d run(s) into %s", len(rows), results_dir)
    return summary
