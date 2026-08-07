"""Command line entry point: ``python -m crisp <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import Config
from .data import load_coherence_set, load_corpus
from .evaluate import evaluate_model, save_results, summarize
from .features import SelectedFeatures, run_selection
from .model import attach_lora, load_model_and_tokenizer
from .report import result_path
from .sae import load_saes
from .train import save_run, train_crisp
from .utils import get_logger, load_dotenv, set_seed, setup_logging

log = get_logger(__name__)


def _record(results: dict, run_name: str, cfg: Config) -> None:
    """Persist one evaluation under artifacts/results and refresh the summary."""
    from . import report

    save_results(results, result_path(run_name, cfg.eval.split))
    report.build()
    print(summarize(results))


def _load_config(args) -> Config:
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.override:
        cfg.apply_overrides(args.override)
    if args.run_name:
        cfg.run_name = args.run_name
    return cfg


def _require_torch(cfg: Config, command: str) -> None:
    """Selection/training read and differentiate the residual stream via hooks."""
    if cfg.model.backend != "torch":
        raise SystemExit(
            f"`crisp {command}` needs the torch backend "
            f"(got model.backend={cfg.model.backend!r}); mlx is inference-only"
        )


def _load_corpora(cfg: Config) -> tuple[list[str], list[str], list[str]]:
    target = load_corpus(
        cfg.data.domain, "target", cfg.data.target_corpus,
        cfg.data.max_target_docs, cfg.data.max_chars, cfg.data.seed,
        cfg.data.target_corpus_repo,
    )
    retain = load_corpus(
        cfg.data.domain, "retain", cfg.data.retain_corpus,
        cfg.data.max_retain_docs, cfg.data.max_chars, cfg.data.seed,
        cfg.data.retain_corpus_repo,
    )
    coherence = load_coherence_set(cfg.data.domain, cfg.data.coherence_path)
    return target, retain, coherence


def _feature_cache_path(cfg: Config) -> Path:
    name = (
        f"{cfg.model.name.replace('/', '_')}__{cfg.data.domain}"
        f"__k{cfg.selection.top_k}_tau{cfg.selection.tau}.json"
    )
    return Path(cfg.selection.cache_dir) / name


def _get_features(cfg, model, tokenizer, saes, target, retain, device, refresh: bool):
    path = _feature_cache_path(cfg)
    if path.exists() and not refresh:
        log.info("loading cached features from %s", path)
        return SelectedFeatures.from_json(path)
    features = run_selection(model, tokenizer, saes, target, retain, cfg, device)
    features.to_json(path)
    log.info("selected %d salient features across %d layers",
             features.total, len(features.layers))
    return features


def cmd_select(args) -> None:
    cfg = _load_config(args)
    _require_torch(cfg, "select")
    set_seed(cfg.data.seed)
    model, tokenizer, device, dtype = load_model_and_tokenizer(cfg)
    saes = load_saes(cfg.sae, cfg.model.sae_layers, model.config.hidden_size, device, dtype)
    target, retain, _ = _load_corpora(cfg)
    features = _get_features(cfg, model, tokenizer, saes, target, retain, device, refresh=True)
    print(json.dumps({str(k): v for k, v in features.layers.items()}, indent=2))


def cmd_train(args) -> None:
    cfg = _load_config(args)
    _require_torch(cfg, "train")
    set_seed(cfg.train.seed)
    model, tokenizer, device, dtype = load_model_and_tokenizer(cfg)
    saes = load_saes(cfg.sae, cfg.model.sae_layers, model.config.hidden_size, device, dtype)
    target, retain, coherence = _load_corpora(cfg)
    features = _get_features(
        cfg, model, tokenizer, saes, target, retain, device, refresh=args.refresh_features
    )

    model = attach_lora(model, cfg)
    history = train_crisp(
        model, tokenizer, saes, features, target, retain, coherence, cfg, device
    )
    save_run(model, tokenizer, cfg, features, history)

    if not args.no_eval:
        results = evaluate_model(
            model, tokenizer, cfg, device,
            judge=not args.no_judge, skip_generation=args.skip_generation,
        )
        _record(results, cfg.run_name, cfg)


def cmd_eval(args) -> None:
    cfg = _load_config(args)
    if args.backend:
        cfg.model.backend = args.backend

    if cfg.model.backend == "mlx":
        from .mlx_backend import load_mlx_model_and_tokenizer

        model, tokenizer, device, _ = load_mlx_model_and_tokenizer(cfg, args.adapter)
    else:
        model, tokenizer, device, _ = load_model_and_tokenizer(cfg)
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
            model.eval()
            log.info("loaded adapter from %s", args.adapter)
    results = evaluate_model(
        model, tokenizer, cfg, device,
        judge=not args.no_judge, skip_generation=args.skip_generation,
    )
    if args.out:
        save_results(results, Path(args.out))
        print(summarize(results))
    else:
        _record(results, cfg.run_name, cfg)


def cmd_baseline(args) -> None:
    from .baselines.elm import ELMConfig, train_elm
    from .baselines.rmu import RMUConfig, train_rmu

    cfg = _load_config(args)
    _require_torch(cfg, "baseline")
    set_seed(cfg.train.seed)
    model, tokenizer, device, _ = load_model_and_tokenizer(cfg)
    target, retain, _ = _load_corpora(cfg)

    if args.method == "rmu":
        rmu_cfg = RMUConfig(
            steps=cfg.train.steps, batch_size=cfg.train.batch_size,
            lr=cfg.train.lr, max_seq_len=cfg.data.max_seq_len, seed=cfg.train.seed,
        )
        if args.baseline_json:
            rmu_cfg.__dict__.update(json.loads(args.baseline_json))
        history = train_rmu(model, tokenizer, target, retain, rmu_cfg, device)
    else:
        model = attach_lora(model, cfg)
        elm_cfg = ELMConfig(
            steps=cfg.train.steps, batch_size=cfg.train.batch_size, lr=cfg.train.lr,
            max_seq_len=cfg.data.max_seq_len, domain=cfg.data.domain, seed=cfg.train.seed,
        )
        if args.baseline_json:
            elm_cfg.__dict__.update(json.loads(args.baseline_json))
        history = train_elm(model, tokenizer, target, retain, elm_cfg, device)

    run_name = f"{cfg.run_name}_{args.method}"
    out = Path(cfg.train.save_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out / "adapter")

    if not args.no_eval:
        results = evaluate_model(
            model, tokenizer, cfg, device,
            judge=not args.no_judge, skip_generation=args.skip_generation,
        )
        _record(results, run_name, cfg)


def cmd_fetch(args) -> None:
    from .fetch import fetch_all

    manifest = fetch_all(
        domains=args.domain or None,
        force=args.force,
        skip_corpora=args.skip_corpora,
        skip_mcq=args.skip_mcq,
    )
    for key, entry in sorted(manifest.items()):
        print(f"{key:28s} {entry['rows']:>7d} rows  {entry['path']}")


def cmd_report(args) -> None:
    from . import report

    summary = report.build(args.results_dir or report.RESULTS_DIR)
    print(report.to_markdown(summary["runs"]))


def cmd_sweep(args) -> None:
    from .sweep import run_sweep

    cfg = _load_config(args)
    _require_torch(cfg, "sweep")
    run_sweep(cfg, n_trials=args.trials, out_dir=args.out or "artifacts/sweeps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crisp", description="CRISP reproduction")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("-c", "--config", help="path to a YAML config")
        p.add_argument("-o", "--override", action="append", default=[],
                       help="section.field=value override (repeatable)")
        p.add_argument("--run-name")

    p_fetch = sub.add_parser("fetch", help="download every dataset into data/")
    p_fetch.add_argument("--domain", action="append", choices=["bio", "cyber"],
                         help="restrict to one domain (repeatable; default: both)")
    p_fetch.add_argument("--force", action="store_true", help="re-download existing files")
    p_fetch.add_argument("--skip-corpora", action="store_true",
                         help="skip the gated forget/retain corpora")
    p_fetch.add_argument("--skip-mcq", action="store_true", help="skip WMDP/MMLU benchmarks")
    p_fetch.set_defaults(func=cmd_fetch)

    p_report = sub.add_parser("report", help="aggregate artifacts/results into a table")
    p_report.add_argument("--results-dir")
    p_report.set_defaults(func=cmd_report)

    p_select = sub.add_parser("select", help="run contrastive feature selection only")
    common(p_select)
    p_select.set_defaults(func=cmd_select)

    p_train = sub.add_parser("train", help="select features and fine-tune with CRISP")
    common(p_train)
    p_train.add_argument("--refresh-features", action="store_true")
    p_train.add_argument("--no-eval", action="store_true")
    p_train.add_argument("--no-judge", action="store_true", help="skip the LLM rater")
    p_train.add_argument("--skip-generation", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="evaluate a base model or a saved adapter")
    common(p_eval)
    p_eval.add_argument("--adapter", help="path to a saved LoRA adapter")
    p_eval.add_argument("--backend", choices=["torch", "mlx"],
                        help="inference backend (default: model.backend, usually torch)")
    p_eval.add_argument("--out")
    p_eval.add_argument("--no-judge", action="store_true")
    p_eval.add_argument("--skip-generation", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_base = sub.add_parser("baseline", help="train an RMU or ELM baseline")
    common(p_base)
    p_base.add_argument("--method", choices=["rmu", "elm"], required=True)
    p_base.add_argument("--baseline-json", help="JSON dict of baseline-specific overrides")
    p_base.add_argument("--no-eval", action="store_true")
    p_base.add_argument("--no-judge", action="store_true")
    p_base.add_argument("--skip-generation", action="store_true")
    p_base.set_defaults(func=cmd_baseline)

    p_sweep = sub.add_parser("sweep", help="hyperparameter search on the validation split")
    common(p_sweep)
    p_sweep.add_argument("--trials", type=int, default=20)
    p_sweep.add_argument("--out")
    p_sweep.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    load_dotenv()  # HF_TOKEN without a manual export
    torch.set_grad_enabled(True)
    args.func(args)


if __name__ == "__main__":
    main()
