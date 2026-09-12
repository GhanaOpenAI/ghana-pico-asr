"""Decoder-only models for recovery, alongside the encoder-decoder path.

NLLB, mT5 and ByT5 are encoder-decoder: source in one side, target out the
other, and the loss only ever sees the target. A causal model like Qwen has one
stream, so the pair has to be laid out as a single sequence and the prompt
masked out of the loss — otherwise the model spends most of its capacity
learning to predict grapheme units, which is the one thing we already have.

The prompt is plain text rather than a chat template: base Qwen2.5 is not
instruction-tuned, and a template it never saw in pretraining is noise.
"""

from __future__ import annotations

PROMPT = "Twi units: {src}\nTwi text:"
IGNORE = -100


def build_example(
    tok,
    source_text: str,
    target_text: str,
    max_source_len: int = 256,
    max_target_len: int = 256,
) -> dict:
    """One (prompt, completion) pair as a single masked sequence.

    Labels are `IGNORE` across the prompt so the loss is computed only on the
    recovered text. The completion keeps its EOS: without one the model has no
    signal for where to stop and generation runs to the token limit.
    """
    # Truncate the *units*, then wrap them, rather than truncating the finished
    # prompt: cutting the prompt to length can remove the trailing "Twi text:"
    # marker, leaving the model no signal for where its answer begins — and
    # `strip_prompt` nothing to split on at inference.
    src_ids = tok(source_text, add_special_tokens=False)["input_ids"][:max_source_len]
    src_trunc = tok.decode(src_ids) if hasattr(tok, "decode") else source_text
    p_ids = tok(PROMPT.format(src=src_trunc), add_special_tokens=False)["input_ids"]
    t_ids = tok(" " + target_text, add_special_tokens=False)["input_ids"][:max_target_len]
    eos = tok.eos_token_id
    if eos is not None:
        t_ids = t_ids + [eos]

    input_ids = p_ids + t_ids
    labels = [IGNORE] * len(p_ids) + list(t_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def collate(features: list[dict], pad_token_id: int) -> dict:
    """Right-pad a batch, padding labels with `IGNORE` rather than the pad id.

    Padding labels with the pad token would train the model to emit padding.
    """
    import torch

    width = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        gap = width - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_token_id] * gap)
        out["attention_mask"].append(f["attention_mask"] + [0] * gap)
        out["labels"].append(f["labels"] + [IGNORE] * gap)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def strip_prompt(decoded: str) -> str:
    """Keep only what the model generated after the prompt's final marker."""
    marker = PROMPT.split("{src}")[-1].strip()  # "Twi text:"
    return decoded.split(marker)[-1].strip() if marker in decoded else decoded.strip()
