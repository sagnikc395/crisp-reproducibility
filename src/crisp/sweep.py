"""Hyperparameter search on the validation split (Appendix F).

The paper runs Bayesian optimisation over 200 configurations per method. We use
Optuna's TPE sampler when it is installed and fall back to random search
otherwise; the search space and selection criterion follow Appendix F.
"""

from __future__ import annotations

import copy
import gc
import json
import random
from pathlib import Path

import torch

from .config import Config
from .data import MMLU_SUBJECTS, load_coherence_set, load_corpus, load_mmlu, load_wmdp_mcq
from .data import load_mmlu_full
from .eval_mcq import evaluate_mcq
from .evaluate import SUBJECT_LABEL
from .features import SelectedFeatures, run_selection
from .metrics import selection_score
from .model import attach_lora, load_model_and_tokenizer
from .sae import load_saes
from .train import train_crisp
from .utils import get_logger, set_seed

log = get_logger(__name__)

SEARCH_SPACE = {
    "top_k": [5, 10, 20, 30, 50],
    "lambda_scale": [10, 20, 30, 40, 50],
    "lora_rank": [4, 8, 16],
    "lr": (1e-5, 1e-4),  # sampled log-uniformly
}


def _validation_metrics(model, tokenizer, cfg: Config, device) -> dict:
    domain = cfg.data.domain
    wmdp = load_wmdp_mcq(domain, split="validation", seed=cfg.data.seed)
    retain = load_mmlu(
        cfg.eval.mmlu_subjects or MMLU_SUBJECTS[domain], split="validation", seed=cfg.data.seed
    )
    # MMLU utility during selection uses the first 10 questions per subject.
    mmlu = load_mmlu_full(max_per_subject=10)
    return {
        "unlearn": evaluate_mcq(model, tokenizer, wmdp, device, SUBJECT_LABEL[domain],
                                cfg.eval.mcq_batch_size, desc="val wmdp")["accuracy"],
        "retain": evaluate_mcq(model, tokenizer, retain, device, SUBJECT_LABEL[domain],
                               cfg.eval.mcq_batch_size, desc="val retain")["accuracy"],
        "mmlu": evaluate_mcq(model, tokenizer, mmlu, device, "various subjects",
                             cfg.eval.mcq_batch_size, desc="val mmlu")["accuracy"],
    }


def _sample(rng: random.Random) -> dict:
    lo, hi = SEARCH_SPACE["lr"]
    import math

    return {
        "top_k": rng.choice(SEARCH_SPACE["top_k"]),
        "lambda_scale": rng.choice(SEARCH_SPACE["lambda_scale"]),
        "lora_rank": rng.choice(SEARCH_SPACE["lora_rank"]),
        "lr": math.exp(rng.uniform(math.log(lo), math.log(hi))),
    }


def _run_trial(base_cfg: Config, params: dict, baseline: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.selection.top_k = params["top_k"]
    cfg.train.lambda_scale = params["lambda_scale"]
    cfg.train.lora_rank = params["lora_rank"]
    cfg.train.lr = params["lr"]

    set_seed(cfg.train.seed)
    model, tokenizer, device, dtype = load_model_and_tokenizer(cfg)
    saes = load_saes(cfg.sae, cfg.model.sae_layers, model.config.hidden_size, device, dtype)
    target = load_corpus(cfg.data.domain, "target", cfg.data.target_corpus,
                         cfg.data.max_target_docs, cfg.data.max_chars, cfg.data.seed,
                         cfg.data.target_corpus_repo)
    retain = load_corpus(cfg.data.domain, "retain", cfg.data.retain_corpus,
                         cfg.data.max_retain_docs, cfg.data.max_chars, cfg.data.seed,
                         cfg.data.retain_corpus_repo)
    coherence = load_coherence_set(cfg.data.domain, cfg.data.coherence_path)

    cache = Path(cfg.selection.cache_dir) / (
        f"{cfg.model.name.replace('/', '_')}__{cfg.data.domain}"
        f"__k{cfg.selection.top_k}_tau{cfg.selection.tau}.json"
    )
    if cache.exists():
        features = SelectedFeatures.from_json(cache)
    else:
        features = run_selection(model, tokenizer, saes, target, retain, cfg, device)
        features.to_json(cache)

    model = attach_lora(model, cfg)
    train_crisp(model, tokenizer, saes, features, target, retain, coherence, cfg, device)
    metrics = _validation_metrics(model, tokenizer, cfg, device)
    score = selection_score(
        metrics["unlearn"], baseline["unlearn"],
        metrics["retain"], baseline["retain"],
        metrics["mmlu"], baseline["mmlu"],
    )

    del model, saes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"params": params, "metrics": metrics, "score": score}


def run_sweep(cfg: Config, n_trials: int = 20, out_dir: str = "artifacts/sweeps") -> list[dict]:
    out = Path(out_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)

    log.info("measuring original-model validation accuracies")
    model, tokenizer, device, _ = load_model_and_tokenizer(cfg)
    baseline = _validation_metrics(model, tokenizer, cfg, device)
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2))
    del model
    gc.collect()
    log.info("original: %s", baseline)

    trials: list[dict] = []
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "top_k": trial.suggest_categorical("top_k", SEARCH_SPACE["top_k"]),
                "lambda_scale": trial.suggest_categorical(
                    "lambda_scale", SEARCH_SPACE["lambda_scale"]
                ),
                "lora_rank": trial.suggest_categorical("lora_rank", SEARCH_SPACE["lora_rank"]),
                "lr": trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True),
            }
            result = _run_trial(cfg, params, baseline)
            trials.append(result)
            (out / "trials.json").write_text(json.dumps(trials, indent=2))
            return result["score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
    except ImportError:
        log.warning("optuna not installed; falling back to random search")
        rng = random.Random(cfg.train.seed)
        for i in range(n_trials):
            result = _run_trial(cfg, _sample(rng), baseline)
            trials.append(result)
            log.info("trial %d score=%.4f params=%s", i, result["score"], result["params"])
            (out / "trials.json").write_text(json.dumps(trials, indent=2))

    best = max(trials, key=lambda t: t["score"]) if trials else {}
    (out / "best.json").write_text(json.dumps(best, indent=2))
    log.info("best config: %s", best.get("params"))
    return trials
