"""Figures for the reproduction, rendered from ``artifacts/results``.

Everything here reads the same result JSONs that ``report.py`` aggregates into
the Table 1 markdown, so the figures and the table can never disagree. Training
curves come from the ``history.json`` each run writes next to its adapter.

Written as PNGs into ``artifacts/figures`` — the one artifacts subdirectory
besides ``results`` that is tracked in git, so a Colab run can commit its
output back to the repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: Colab, CI, and `crisp plots` from a terminal

import matplotlib.pyplot as plt  # noqa: E402

from .report import COLUMNS, METHOD_ORDER, RESULTS_DIR, collect  # noqa: E402
from .utils import REPO_ROOT, get_logger  # noqa: E402

log = get_logger(__name__)

FIGURES_DIR = REPO_ROOT / "artifacts" / "figures"
RUNS_GLOB = "artifacts/runs*/*/history.json"

#: One colour per method, kept constant across every figure.
METHOD_COLORS = {
    "original": "#8c8c8c",
    "crisp": "#1f77b4",
    "rmu": "#d95f02",
    "elm": "#7570b3",
}
DEFAULT_COLOR = "#444444"


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, DEFAULT_COLOR)


def _methods_in(rows: list[dict]) -> list[dict]:
    """Rows of one config, in the canonical original → crisp → rmu → elm order."""
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda r: rank.get(r["method"], len(METHOD_ORDER)))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def metric_bars(rows: list[dict], config: str, out_dir: Path) -> Path | None:
    """Grouped bars: every Table 1 column, one bar group per method.

    Columns are on a shared 0-100 axis, which is what makes the grouping
    readable; ``Concept`` and ``Fluency`` are 0-2 rater scores, so they are
    rescaled by 50 (the same scaling Eq. 12 applies) and labelled as such.
    """
    rows = _methods_in(rows)
    if not rows:
        return None

    scale = {"fluency": 50.0, "concept": 50.0}
    labels = [
        f"{label} (x50)" if key in scale else label
        for key, label, _ in COLUMNS
    ]
    width = 0.8 / len(rows)

    fig, ax = plt.subplots(figsize=(1.6 * len(COLUMNS), 4.2))
    for i, row in enumerate(rows):
        xs, ys = [], []
        for j, (key, _, _) in enumerate(COLUMNS):
            if row.get(key) is None:
                continue  # --no-judge leaves the generation columns empty
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(row[key] * scale.get(key, 1.0))
        ax.bar(xs, ys, width=width, label=row["method"], color=_color(row["method"]))

    ax.set_xticks(range(len(COLUMNS)), labels, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 100)
    ax.set_title(f"{config} — WMDP down is better, everything else up")
    ax.legend(frameon=False, ncols=len(rows))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    return _save(fig, out_dir / f"metrics_{config}.png")


def unlearning_tradeoff(rows: list[dict], config: str, out_dir: Path) -> Path | None:
    """The paper's actual claim: WMDP drops without dragging utility down with it.

    x is the forget axis (WMDP accuracy, lower is better), y the retain axis
    (in-domain MMLU). The bottom-right corner is where a good method lands.
    """
    rows = [r for r in _methods_in(rows)
            if r.get("unlearn_acc") is not None and r.get("retain_acc") is not None]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for row in rows:
        # alpha so methods that land on the same point stay distinguishable
        ax.scatter(row["unlearn_acc"], row["retain_acc"], s=140, alpha=0.75,
                   color=_color(row["method"]), zorder=3)
        ax.annotate(row["method"], (row["unlearn_acc"], row["retain_acc"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    # 25% is chance on 4-way WMDP MCQs: the floor unlearning can reach.
    ax.axvline(25.0, ls="--", lw=1, color="#999999")
    ax.annotate("chance (25%)", (25.0, ax.get_ylim()[0]), rotation=90,
                textcoords="offset points", xytext=(4, 6), fontsize=8, color="#666666")
    ax.set_xlabel("WMDP accuracy (lower = more unlearning)")
    ax.set_ylabel("in-domain MMLU (higher = more retained)")
    ax.set_title(f"{config} — forget/retain trade-off")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    return _save(fig, out_dir / f"tradeoff_{config}.png")


def training_curves(root: Path, out_dir: Path) -> list[Path]:
    """Loss curves per run, with the CRISP terms of Eq. 11 broken out.

    The three terms live on very different scales (the retention term starts at
    exactly 0 because the adapter starts as a no-op), so each gets its own
    panel rather than being crammed onto one axis.
    """
    written: list[Path] = []
    for history_path in sorted(root.glob(RUNS_GLOB)):
        try:
            payload = json.loads(history_path.read_text())
        except json.JSONDecodeError:
            log.warning("skipping unreadable history %s", history_path)
            continue
        history = payload.get("history") if isinstance(payload, dict) else payload
        if not history:
            continue

        run = history_path.parent.name
        steps = [h["step"] for h in history]
        terms = [k for k in ("loss", "unlearn", "retain", "coherence")
                 if any(k in h for h in history)]

        fig, axes = plt.subplots(1, len(terms), figsize=(3.4 * len(terms), 3.0),
                                 squeeze=False)
        for ax, term in zip(axes[0], terms):
            ax.plot(steps, [h.get(term) for h in history], color=_color(run.split("_")[-1]))
            ax.set_title(term)
            ax.set_xlabel("step")
            ax.grid(alpha=0.25)
            ax.set_axisbelow(True)
        fig.suptitle(f"{run} — training loss")
        written.append(_save(fig, out_dir / f"training_{run}.png"))
    return written


def build(results_dir: Path | str = RESULTS_DIR,
          out_dir: Path | str = FIGURES_DIR,
          root: Path | str = REPO_ROOT) -> list[Path]:
    """Render every figure the current artifacts support. Returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(results_dir)

    written: list[Path] = []
    by_config: dict[str, list[dict]] = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)
    for config, config_rows in sorted(by_config.items()):
        for path in (metric_bars(config_rows, config, out_dir),
                     unlearning_tradeoff(config_rows, config, out_dir)):
            if path is not None:
                written.append(path)
    written += training_curves(Path(root), out_dir)

    (out_dir / "README.md").write_text(_index(written, out_dir))
    log.info("rendered %d figure(s) into %s", len(written), out_dir)
    return written


def _index(paths: list[Path], out_dir: Path) -> str:
    if not paths:
        return "# Figures\n\nNo results yet. Run `scripts/reproduce.sh <config.yaml>` first.\n"
    lines = ["# Figures", "",
             "Regenerate with `python -m crisp plots` after any new evaluation.", ""]
    for path in paths:
        lines += [f"### {path.stem}", "", f"![{path.stem}]({path.relative_to(out_dir)})", ""]
    return "\n".join(lines)
