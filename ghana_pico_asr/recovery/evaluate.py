"""Scoring the recovery stage.

Unit error rate measures stage 1. Stage 2 produces words, so it is scored on
**character and word error rate against the reference text** — plus the share
of utterances recovered exactly, which is the number a reader of the pipeline's
output actually feels.

The baseline to beat is not zero. Joining the units into a string already
produces something Twi-shaped, so a model that merely inserts spaces would
score respectably on CER; `baseline_cer` reports that, and a model is only
earning its keep when it is well clear of it.
"""

from __future__ import annotations

import torch

from ..devset import edit_distance


def cer(hyp: str, ref: str) -> float:
    if not ref:
        return 1.0 if hyp else 0.0
    return edit_distance(list(hyp), list(ref)) / len(ref)


def wer(hyp: str, ref: str) -> float:
    r = ref.split()
    if not r:
        return 1.0 if hyp.split() else 0.0
    return edit_distance(hyp.split(), r) / len(r)


@torch.no_grad()
def score_model(
    model,
    tok,
    rows: list[dict],
    lang_code: str | None,
    batch_size: int = 16,
    max_source_len: int = 192,
    max_target_len: int = 160,
    num_beams: int = 4,
) -> dict:
    """Generate for `rows` and report CER/WER against the reference text."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    # None for the T5 family, which has no language token to open with.
    forced_bos = tok.convert_tokens_to_ids(lang_code) if lang_code else None

    hyps: list[str] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        enc = tok(
            [r["source_text"] for r in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_len,
        ).to(device)
        gen_kw = {"max_new_tokens": max_target_len, "num_beams": num_beams}
        if forced_bos is not None:
            gen_kw["forced_bos_token_id"] = forced_bos
        out = model.generate(**enc, **gen_kw)
        hyps.extend(tok.batch_decode(out, skip_special_tokens=True))

    refs = [r["target_text"] for r in rows]
    srcs = [r["source_text"] for r in rows]
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "cer": sum(cer(h, r) for h, r in zip(hyps, refs)) / n,
        "wer": sum(wer(h, r) for h, r in zip(hyps, refs)) / n,
        "exact_match": sum(h.strip() == r.strip() for h, r in zip(hyps, refs)) / n,
        # What the raw units already score, before the model does anything.
        "baseline_cer": sum(cer(s, r) for s, r in zip(srcs, refs)) / n,
        "samples": [
            {"src": s[:80], "hyp": h[:80], "ref": r[:80]}
            for s, h, r in list(zip(srcs, hyps, refs))[:5]
        ],
    }
