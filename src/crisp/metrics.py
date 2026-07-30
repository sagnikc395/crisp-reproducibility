"""Score aggregation (Section 4.4 Eq. 12, Appendix F selection criteria)."""

from __future__ import annotations

import math

from .utils import harmonic_mean


def overall_score(
    unlearn_acc: float,
    retain_acc: float,
    mmlu_acc: float,
    fluency: float,
    concept: float,
) -> float:
    """Eq. 12: ``HM(100 - U, R, M, F * 50, C * 50)``.

    Fluency and concept are 0-2 scales normalised to 0-100.
    """
    return harmonic_mean([100.0 - unlearn_acc, retain_acc, mmlu_acc, fluency * 50.0, concept * 50.0])


def relative_change(edited: float, original: float) -> float:
    """(A_edit - A_orig) / A_orig (Appendix F)."""
    return (edited - original) / original if original else 0.0


def selection_score(
    unlearn_edit: float,
    unlearn_orig: float,
    retain_edit: float,
    retain_orig: float,
    mmlu_edit: float,
    mmlu_orig: float,
) -> float:
    """Geometric mean of unlearning, retention and general-capability scores,
    used to pick a hyperparameter configuration on the validation split."""
    unlearning = 1.0 - relative_change(unlearn_edit, unlearn_orig)
    retention = 1.0 + relative_change(retain_edit, retain_orig)
    general = 1.0 + relative_change(mmlu_edit, mmlu_orig)
    parts = [max(p, 0.0) for p in (unlearning, retention, general)]
    return math.prod(parts) ** (1 / 3)


def format_table(rows: list[dict], columns: list[str]) -> str:
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in columns}
    lines = [" | ".join(c.ljust(widths[c]) for c in columns)]
    lines.append("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        lines.append(" | ".join(f"{row.get(c, '')}".ljust(widths[c]) for c in columns))
    return "\n".join(lines)
