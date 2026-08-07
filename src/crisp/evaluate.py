"""End-to-end evaluation harness for a (possibly unlearned) model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import Config
from .data import (
    CONCEPT_NAME,
    MMLU_SUBJECTS,
    load_gen_prompts,
    load_mmlu,
    load_mmlu_full,
    load_wmdp_mcq,
)
from .eval_gen import generate_continuations, judge_generations, unload_judge
from .eval_mcq import evaluate_mcq
from .metrics import overall_score
from .utils import get_logger

log = get_logger(__name__)

SUBJECT_LABEL = {"bio": "biology", "cyber": "computer security"}


def evaluate_model(
    model,
    tokenizer,
    cfg: Config,
    device: torch.device,
    judge: bool = True,
    skip_generation: bool = False,
) -> dict:
    domain = cfg.data.domain
    split = cfg.eval.split
    seed = cfg.data.seed
    results: dict = {"domain": domain, "split": split}

    # Unlearning accuracy on the held-out WMDP MCQs (lower is better).
    wmdp = load_wmdp_mcq(domain, split=split, seed=seed)
    results["unlearn_acc"] = evaluate_mcq(
        model, tokenizer, wmdp, device, SUBJECT_LABEL[domain],
        cfg.eval.mcq_batch_size, desc="wmdp",
    )["accuracy"]

    # In-domain retention on the matching MMLU subjects.
    subjects = cfg.eval.mmlu_subjects or MMLU_SUBJECTS[domain]
    retain_items = load_mmlu(subjects, split=split, seed=seed)
    results["retain_acc"] = evaluate_mcq(
        model, tokenizer, retain_items, device, SUBJECT_LABEL[domain],
        cfg.eval.mcq_batch_size, desc="retain",
    )["accuracy"]

    # General capability on MMLU.
    mmlu_items = load_mmlu_full(max_per_subject=cfg.eval.mmlu_max_per_subject or 0)
    results["mmlu"] = evaluate_mcq(
        model, tokenizer, mmlu_items, device, "various subjects",
        cfg.eval.mcq_batch_size, desc="mmlu",
    )["accuracy"]

    # Generation quality (AxBench-style fluency / concept scores).
    if not skip_generation:
        prefixes = load_gen_prompts(domain, cfg.eval.gen_prompts_path)
        generations = generate_continuations(
            model, tokenizer, prefixes, device, cfg.eval.gen_max_new_tokens
        )
        results["generations"] = [{"prefix": g.prefix, "text": g.text} for g in generations]
        if judge:
            concept = cfg.eval.judge_concept or CONCEPT_NAME[domain]
            scored = judge_generations(
                generations,
                concept,
                model=cfg.eval.judge_model,
                max_new_tokens=cfg.eval.judge_max_new_tokens,
                batch_size=cfg.eval.judge_batch_size,
                device=cfg.eval.judge_device,
                dtype=cfg.eval.judge_dtype,
                seed=cfg.eval.judge_seed,
            )
            unload_judge()  # the rater and the model under test share the device
            results.update(
                fluency=scored["fluency"],
                fluency_std=scored["fluency_std"],
                concept=scored["concept"],
                concept_std=scored["concept_std"],
                judged=scored["rows"],
            )

    if all(k in results for k in ("fluency", "concept")):
        results["overall"] = overall_score(
            results["unlearn_acc"],
            results["retain_acc"],
            results["mmlu"],
            results["fluency"],
            results["concept"],
        )
    return results


def save_results(results: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    log.info("wrote results to %s", path)
    return path


def summarize(results: dict) -> str:
    keys = ["overall", "unlearn_acc", "retain_acc", "mmlu", "fluency", "concept"]
    parts = [f"{k}={results[k]:.2f}" for k in keys if isinstance(results.get(k), float)]
    return "  ".join(parts)
