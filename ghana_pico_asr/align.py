"""CTC forced alignment of Twi 2-char grapheme units.

Wraps `MahmoudAshraf97/ctc-forced-aligner` but bypasses its text front-end:
``preprocess_text`` only offers sentence/word/char splitting, so we build the
paired ``(tokens_starred, text_starred)`` lists ourselves from
:mod:`ghana_pico_asr.languages`. Everything downstream of that — emissions,
Viterbi forced alignment, span padding — is the library's own code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import config as C
from .languages import Language, build_tokens_starred, get_language


class AlignmentError(RuntimeError):
    """Raised for an utterance that cannot be aligned; the caller skips it."""


@dataclass
class UnitSpan:
    unit: str
    start: float  # seconds
    end: float  # seconds
    score: float  # mean CTC log-probability over the span


def load_aligner(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    lang: Language | None = None,
):
    """Load the MMS forced-alignment model and validate its vocabulary."""
    from ctc_forced_aligner import load_alignment_model

    model, tokenizer = load_alignment_model(device, C.ALIGNER_MODEL, dtype=dtype)

    # Mirror get_alignments(): it lower-cases the vocab and appends <star>
    # itself, so <star> is deliberately absent from the raw vocabulary.
    lang = lang or get_language()
    vocab = {k.lower(): v for k, v in tokenizer.get_vocab().items()}
    missing = sorted(
        {ch for unit in lang.units for ch in lang.romanize_unit(unit) if ch not in vocab}
    )
    if missing:
        # get_alignments() silently drops unknown characters, which then trips
        # get_spans()'s assertion with an unreadable error. Fail clearly here.
        raise AlignmentError(f"aligner vocabulary is missing characters: {missing}")

    return model, tokenizer


def _postprocess(
    text_starred: list[str],
    spans: list,
    stride_ms: float,
    scores: np.ndarray,
) -> list[UnitSpan]:
    """Turn frame spans into second-valued :class:`UnitSpan`s.

    Reimplements the library's ``postprocess_results`` so we can keep the
    per-unit score and skip its word-level segment merging.
    """
    out: list[UnitSpan] = []
    for i, text in enumerate(text_starred):
        if text == "<star>":
            continue
        span = spans[i]
        first = span[0].start
        last = span[-1].end + 1
        out.append(
            UnitSpan(
                unit=text,
                start=first * stride_ms / 1000.0,
                end=last * stride_ms / 1000.0,
                score=float(scores[first:last].mean()),
            )
        )
    return out


@torch.inference_mode()
def align_utterance(
    model,
    tokenizer,
    waveform: np.ndarray,
    text: str,
    batch_size: int = 8,
    lang: Language | None = None,
) -> list[UnitSpan]:
    """Align ``text`` against ``waveform`` (mono float32 @16 kHz).

    Returns one :class:`UnitSpan` per grapheme unit, in order.
    """
    from ctc_forced_aligner import generate_emissions, get_alignments, get_spans

    lang = lang or get_language()
    words = lang.segment(text)
    if not words:
        raise AlignmentError("no alignable grapheme units in transcript")

    tokens_starred, text_starred = build_tokens_starred(lang, words)
    n_units = sum(len(w) for w in words)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    audio = torch.from_numpy(waveform).to(device=device, dtype=dtype)

    emissions, stride_ms = generate_emissions(model, audio, batch_size=batch_size)

    # CTC forced alignment needs T >= L + N_repeat: one frame per target plus a
    # separating blank between adjacent identical targets, where L counts the
    # <star> wildcards too. Under-checking this overruns the C++ Viterbi buffer
    # rather than failing cleanly.
    target_chars: list[str] = []
    for tok in tokens_starred:
        target_chars += tok.split(" ") if tok != "<star>" else ["<star>"]
    n_targets = len(target_chars)
    n_repeat = sum(
        1 for i in range(1, n_targets) if target_chars[i] == target_chars[i - 1]
    )
    needed = n_targets + n_repeat
    if emissions.shape[0] < needed:
        raise AlignmentError(
            f"audio too short for transcript: {emissions.shape[0]} frames < "
            f"{needed} required ({n_targets} targets + {n_repeat} repeats) "
            f"for {n_units} units in {len(words)} words"
        )

    try:
        segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank)
    except (AssertionError, RuntimeError, IndexError) as exc:
        raise AlignmentError(f"forced alignment failed: {exc}") from exc

    units = _postprocess(text_starred, spans, stride_ms, scores.squeeze())
    if len(units) != n_units:
        raise AlignmentError(f"expected {n_units} spans, got {len(units)}")
    return units
