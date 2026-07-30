"""ELM baseline (Gandikota et al., 2024 -- "Erasing Conceptual Knowledge from LMs").

Reimplemented from the paper description: the model is fine-tuned (LoRA on early
layers) so that its next-token distribution on the target corpus matches a
classifier-free-guidance style *erased* distribution

    log p_target = log p_0(x) + eta * [ log p_0(x | benign) - log p_0(x | concept) ]

while a retention term keeps the distribution on the retain corpus unchanged and
a fluency term keeps generations well-formed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from ..model import base_model, tokenize_batch
from ..utils import get_logger, sample_batches, set_seed

log = get_logger(__name__)

CONCEPT_PROMPT = {
    "bio": "The following text is about hazardous biosecurity and bioweapons knowledge.\n",
    "cyber": "The following text is about hazardous offensive cybersecurity knowledge.\n",
}
BENIGN_PROMPT = {
    "bio": "The following text is about general, harmless introductory biology.\n",
    "cyber": "The following text is about general, harmless computer science.\n",
}


@dataclass
class ELMConfig:
    eta: float = 1500.0  # guidance strength
    erase_scale: float = 1.0
    retain_scale: float = 1.0
    fluency_scale: float = 1.0
    lr: float = 1.1e-5
    lora_rank: int = 4
    lora_alpha: int = 8
    steps: int = 200
    batch_size: int = 2
    max_seq_len: int = 256
    domain: str = "bio"
    seed: int = 0


def _conditioned_logits(model, tokenizer, texts, prompt, cfg, device):
    """log p_0(x_t | x_<t, prompt) aligned to the unconditioned token positions."""
    batch = tokenize_batch(tokenizer, [prompt + t for t in texts], cfg.max_seq_len, device)
    with torch.no_grad(), base_model(model):
        logits = model(**batch).logits.float()
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return F.log_softmax(logits[:, prompt_len:, :], dim=-1), prompt_len


def train_elm(
    model,
    tokenizer,
    target_docs: list[str],
    retain_docs: list[str],
    cfg: ELMConfig,
    device: torch.device,
) -> dict:
    set_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("ELM expects a LoRA-wrapped model with trainable parameters")
    optimizer = AdamW(params, lr=cfg.lr)

    concept_prompt = CONCEPT_PROMPT[cfg.domain]
    benign_prompt = BENIGN_PROMPT[cfg.domain]
    target_batches = sample_batches(target_docs, cfg.batch_size, cfg.steps, cfg.seed)
    retain_batches = sample_batches(retain_docs, cfg.batch_size, cfg.steps, cfg.seed + 1)

    history = []
    model.train()
    start = time.time()

    for step in tqdm(range(cfg.steps), desc="elm"):
        optimizer.zero_grad(set_to_none=True)

        # --- Erasure ---------------------------------------------------------
        texts = next(target_batches)
        batch = tokenize_batch(tokenizer, texts, cfg.max_seq_len, device)
        mask = batch["attention_mask"].bool()
        with torch.no_grad(), base_model(model):
            base_logp = F.log_softmax(model(**batch).logits.float(), dim=-1)
        concept_logp, _ = _conditioned_logits(
            model, tokenizer, texts, concept_prompt, cfg, device
        )
        benign_logp, _ = _conditioned_logits(
            model, tokenizer, texts, benign_prompt, cfg, device
        )
        length = min(base_logp.shape[1], concept_logp.shape[1], benign_logp.shape[1])
        target_logp = (
            base_logp[:, :length]
            + cfg.eta * (benign_logp[:, :length] - concept_logp[:, :length])
        )
        target_probs = F.softmax(target_logp, dim=-1)

        logits = model(**batch).logits.float()[:, :length]
        token_mask = mask[:, :length]
        l_erase = -(target_probs * F.log_softmax(logits, dim=-1)).sum(-1)[token_mask].mean()

        # --- Retention -------------------------------------------------------
        batch = tokenize_batch(tokenizer, next(retain_batches), cfg.max_seq_len, device)
        mask = batch["attention_mask"].bool()
        with torch.no_grad(), base_model(model):
            ref_logp = F.log_softmax(model(**batch).logits.float(), dim=-1)
        logp = F.log_softmax(model(**batch).logits.float(), dim=-1)
        l_retain = F.kl_div(
            logp[mask], ref_logp[mask], log_target=True, reduction="batchmean"
        )

        # --- Fluency: stay close to the original model's argmax on the target -
        with torch.no_grad(), base_model(model):
            batch_t = tokenize_batch(tokenizer, texts, cfg.max_seq_len, device)
            ref_tokens = model(**batch_t).logits.float().argmax(-1)
        logits_t = model(**batch_t).logits.float()
        t_mask = batch_t["attention_mask"].bool()
        l_fluency = F.cross_entropy(logits_t[t_mask], ref_tokens[t_mask])

        loss = (
            cfg.erase_scale * l_erase
            + cfg.retain_scale * l_retain
            + cfg.fluency_scale * l_fluency
        )
        loss.backward()
        optimizer.step()

        history.append(
            {"step": step, "loss": float(loss.detach()), "erase": float(l_erase.detach()),
             "retain": float(l_retain.detach()), "fluency": float(l_fluency.detach())}
        )
        if (step + 1) % 20 == 0:
            log.info("elm step %d | loss %.4f", step + 1, history[-1]["loss"])

    model.eval()
    return {"history": history, "seconds": time.time() - start}
