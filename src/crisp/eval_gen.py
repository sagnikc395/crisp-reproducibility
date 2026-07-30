"""Fluency and Concept scores via an LLM rater (Appendix E)."""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from .utils import chunked, get_logger

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


def _parse_rating(response: str) -> int | None:
    match = _RATING.search(response)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"\b([0-2])\b", response[-40:])
    return int(fallback[-1]) if fallback else None


def judge_generations(
    generations: list[Generation],
    concept: str,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 400,
) -> dict:
    """Score each generation for fluency and concept presence with Claude."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; rerun with --no-judge to only dump generations"
        )
    import anthropic

    client = anthropic.Anthropic()
    fluency: list[int] = []
    concept_scores: list[int] = []
    rows = []

    for gen in tqdm(generations, desc="judge", leave=False):
        def ask(system: str, user: str) -> int | None:
            reply = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in reply.content if b.type == "text")
            return _parse_rating(text)

        f = ask(FLUENCY_SYSTEM, FLUENCY_USER.format(prefix=gen.prefix, generated_text=gen.text))
        c = ask(
            CONCEPT_SYSTEM,
            CONCEPT_USER.format(concept=concept, prefix=gen.prefix, generated_text=gen.text),
        )
        if f is not None:
            fluency.append(f)
        if c is not None:
            concept_scores.append(c)
        rows.append({"prefix": gen.prefix, "text": gen.text, "fluency": f, "concept": c})

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
