"""Fluency and Concept scores via a local LLM rater (Appendix E)."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from .utils import chunked, get_logger, hf_token, resolve_device

log = get_logger(__name__)

FLUENCY_SYSTEM = """Please act as an impartial judge and evaluate the fluency of the generated text provided below. You will be given a prefix and the text generated from that prefix. Focus your evaluation solely on the generated text, not the prefix. Assess only fluency, disregarding completeness, relevance, coherence with any broader context, or informativeness. Begin your evaluation by briefly describing the fluency of the generated text, noting any unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that may hinder readability. After providing your explanation, rate the generated text's fluency on a scale from 0 to 2, where 0 indicates the text is not fluent and highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the text is fluent and almost perfect. Provide your rating using this exact format: "Rating: [[score]]"."""

FLUENCY_USER = """[Prefix Start]
{prefix}
[Prefix End]
[Generated Text Start]
{generated_text}
[Generated Text End]"""

CONCEPT_SYSTEM = """Please act as an impartial judge and evaluate whether the specified concept is presented in the generated text provided below. You will be given a prefix and the text generated from that prefix. Focus your evaluation solely on the generated text, not the prefix. Assess only whether the concept is clearly incorporated, without regard for grammar or logical coherence. Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated in the generated text. Point out references to the concept in the generated text, noting any minimal or awkward incorporations. After providing your explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with stronger and more natural integration. Provide your rating using this exact format: "Rating: [[score]]"."""

CONCEPT_USER = """[Concept Start]
{concept}
[Concept End]
[Prefix Start]
{prefix}
[Prefix End]
[Generated Text Start]
{generated_text}
[Generated Text End]"""

_RATING = re.compile(r"Rating:\s*\[\[\s*([0-2])\s*\]\]")


@dataclass
class Generation:
    prefix: str
    text: str


@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    prefixes: list[str],
    device: torch.device,
    max_new_tokens: int = 50,
    batch_size: int = 8,
) -> list[Generation]:
    """Greedy decoding, max 50 new tokens (Appendix E.2)."""
    outputs: list[Generation] = []
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        for batch in tqdm(list(chunked(prefixes, batch_size)), desc="generate", leave=False):
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend(Generation(p, t.strip()) for p, t in zip(batch, texts))
    finally:
        tokenizer.padding_side = original_side
    return outputs


def _strip_thinking(response: str) -> str | None:
    """Return the answer after the reasoning block, or None if it never closed.

    Qwen3 ``-Thinking`` checkpoints always open a ``<think>`` block and only
    emit the rating after ``</think>``. A response with no closing tag was cut
    off mid-reasoning, so it holds no rating at all -- returning the raw text
    would let the digit fallback below score the model on its own scratchpad.
    """
    if "</think>" not in response:
        return None if "<think>" in response else response
    return response.rsplit("</think>", 1)[1]


def _parse_rating(response: str) -> int | None:
    answer = _strip_thinking(response)
    if answer is None:
        return None
    match = _RATING.search(answer)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"\b([0-2])\b", answer[-40:])
    return int(fallback[-1]) if fallback else None


_JUDGE_CACHE: dict[str, tuple] = {}


def load_judge(model: str, device: str | None = None, dtype: str = "bfloat16"):
    """Load (and memoise) the rater. Cached so a sweep pays the load once."""
    key = f"{model}|{device}|{dtype}"
    if key not in _JUDGE_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = resolve_device(device)
        # Inference only, so bf16 is safe on MPS here -- the fp32 default in
        # resolve_dtype exists for optimiser stability during training.
        torch_dtype = getattr(torch, dtype)
        tok = AutoTokenizer.from_pretrained(model, token=hf_token())
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        judge = AutoModelForCausalLM.from_pretrained(model, dtype=torch_dtype, token=hf_token())
        judge.to(resolved)
        judge.eval()
        log.info("loaded judge %s on %s (%s)", model, resolved, torch_dtype)
        _JUDGE_CACHE[key] = (judge, tok, resolved)
    return _JUDGE_CACHE[key]


def unload_judge() -> None:
    """Drop the cached rater so the model under test gets the memory back."""
    _JUDGE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def _rate(judge, tok, device, prompts: list[tuple[str, str]], max_new_tokens: int,
          batch_size: int, seed: int) -> list[int | None]:
    """Run (system, user) prompt pairs through the rater and parse the scores."""
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for system, user in prompts
    ]
    original_side = tok.padding_side
    tok.padding_side = "left"
    scores: list[int | None] = []
    try:
        for batch in tqdm(list(chunked(texts, batch_size)), desc="judge", leave=False):
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            # Qwen3 thinking checkpoints degenerate into repetition under greedy
            # decoding, so use the sampling settings from the model card and fix
            # the seed instead to keep runs reproducible.
            torch.manual_seed(seed)
            out = judge.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                pad_token_id=tok.pad_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            for reply in tok.batch_decode(new_tokens, skip_special_tokens=False):
                scores.append(_parse_rating(reply))
    finally:
        tok.padding_side = original_side
    return scores


def judge_generations(
    generations: list[Generation],
    concept: str,
    model: str = "Qwen/Qwen3-4B-Thinking-2507",
    max_new_tokens: int = 2048,
    batch_size: int = 4,
    device: str | None = None,
    dtype: str = "bfloat16",
    seed: int = 0,
) -> dict:
    """Score each generation for fluency and concept presence with a local rater."""
    judge, tok, resolved = load_judge(model, device, dtype)

    fluency_prompts = [
        (FLUENCY_SYSTEM, FLUENCY_USER.format(prefix=g.prefix, generated_text=g.text))
        for g in generations
    ]
    concept_prompts = [
        (CONCEPT_SYSTEM,
         CONCEPT_USER.format(concept=concept, prefix=g.prefix, generated_text=g.text))
        for g in generations
    ]
    f_scores = _rate(judge, tok, resolved, fluency_prompts, max_new_tokens, batch_size, seed)
    c_scores = _rate(judge, tok, resolved, concept_prompts, max_new_tokens, batch_size, seed)

    fluency = [s for s in f_scores if s is not None]
    concept_scores = [s for s in c_scores if s is not None]
    unparsed = sum(s is None for s in f_scores + c_scores)
    if unparsed:
        log.warning(
            "%d/%d ratings unparsed (raise eval.judge_max_new_tokens if the rater "
            "is being truncated mid-reasoning)", unparsed, len(f_scores) + len(c_scores)
        )
    rows = [
        {"prefix": g.prefix, "text": g.text, "fluency": f, "concept": c}
        for g, f, c in zip(generations, f_scores, c_scores)
    ]

    def summarise(values: list[int]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)

    f_mean, f_std = summarise(fluency)
    c_mean, c_std = summarise(concept_scores)
    return {
        "fluency": f_mean,
        "fluency_std": f_std,
        "concept": c_mean,
        "concept_std": c_std,
        "n": len(rows),
        "rows": rows,
    }
