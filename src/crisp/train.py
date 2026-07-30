"""CRISP fine-tuning loop (Section 3.3)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import Config
from .features import SelectedFeatures
from .losses import representation_distance, total_loss, unlearning_loss
from .model import ResidualCapture, base_model, decoder_layers, tokenize_batch
from .sae import SAEBundle
from .utils import get_logger, sample_batches, set_seed

log = get_logger(__name__)


def _reference_acts(model, batch, layers, ) -> dict[int, torch.Tensor]:
    """Hidden states of the original frozen model M0 (LoRA adapters disabled)."""
    with torch.no_grad(), base_model(model), ResidualCapture(model, layers) as capture:
        model(**batch)
        return {layer: capture.acts[layer].detach() for layer in layers}


def train_crisp(
    model,
    tokenizer,
    saes: SAEBundle,
    features: SelectedFeatures,
    target_docs: list[str],
    retain_docs: list[str],
    coherence_docs: list[str],
    cfg: Config,
    device: torch.device,
) -> dict:
    """Optimise Eq. 11 with LoRA and return the training history."""
    tc = cfg.train
    set_seed(tc.seed)

    sae_layers = saes.layers
    final_layer = len(decoder_layers(model)) - 1  # coherence uses the last layer
    alpha, beta, gamma = tc.resolved_alpha, tc.beta, tc.gamma

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=tc.lr)
    scheduler = None
    if tc.warmup_steps:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / tc.warmup_steps)
        )

    target_batches = sample_batches(target_docs, tc.batch_size, tc.steps, tc.seed)
    retain_batches = sample_batches(retain_docs, tc.batch_size, tc.steps, tc.seed + 1)
    coherence_batches = sample_batches(coherence_docs, tc.batch_size, tc.steps, tc.seed + 2)

    history: list[dict] = []
    model.train()
    start = time.time()

    progress = tqdm(range(tc.steps), desc="crisp")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)

        # --- Unlearning term (Eq. 9) on the target corpus -------------------
        batch = tokenize_batch(
            tokenizer, next(target_batches), cfg.data.max_seq_len, device
        )
        mask = batch["attention_mask"].bool()
        with ResidualCapture(model, sae_layers) as capture:
            model(**batch)
            l_unlearn = unlearning_loss(
                {l: capture.acts[l] for l in sae_layers},
                mask,
                saes,
                features.layers,
                tc.lambda_scale,
            )

        # --- Retention term (Eq. 10) on the retain corpus -------------------
        batch = tokenize_batch(
            tokenizer, next(retain_batches), cfg.data.max_seq_len, device
        )
        mask = batch["attention_mask"].bool()
        ref = _reference_acts(model, batch, sae_layers)
        with ResidualCapture(model, sae_layers) as capture:
            model(**batch)
            l_retain = representation_distance(
                {l: capture.acts[l] for l in sae_layers}, ref, mask, tc.retain_reduction
            )
        del ref

        # --- Coherence term on the curated set, final layer only ------------
        batch = tokenize_batch(
            tokenizer, next(coherence_batches), cfg.data.max_seq_len, device
        )
        mask = batch["attention_mask"].bool()
        ref = _reference_acts(model, batch, [final_layer])
        with ResidualCapture(model, [final_layer]) as capture:
            model(**batch)
            l_coherence = representation_distance(
                {final_layer: capture.acts[final_layer]}, ref, mask, tc.retain_reduction
            )
        del ref

        loss = total_loss(l_unlearn, l_retain, l_coherence, alpha, beta, gamma)
        loss.backward()
        if tc.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        record = {
            "step": step,
            "loss": float(loss.detach()),
            "unlearn": float(l_unlearn.detach()),
            "retain": float(l_retain.detach()),
            "coherence": float(l_coherence.detach()),
        }
        history.append(record)
        progress.set_postfix(
            loss=f"{record['loss']:.3f}",
            unl=f"{record['unlearn']:.2f}",
            ret=f"{record['retain']:.2f}",
        )
        if (step + 1) % tc.log_every == 0:
            log.info(
                "step %4d | total %.4f | unlearn %.4f | retain %.4f | coherence %.4f",
                step + 1, record["loss"], record["unlearn"],
                record["retain"], record["coherence"],
            )

    model.eval()
    log.info("training finished in %.1fs", time.time() - start)
    return {"history": history, "seconds": time.time() - start}


def save_run(model, tokenizer, cfg: Config, features: SelectedFeatures, history: dict) -> Path:
    out = Path(cfg.train.save_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    features.to_json(out / "features.json")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    log.info("saved run to %s", out)
    return out
