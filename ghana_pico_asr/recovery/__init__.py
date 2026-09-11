"""Stage 2: grapheme units -> Twi text.

The recogniser emits sounds, not words. This turns `ɔ y ɛ n e h o` into
"Ɔyɛ ne ho" by fine-tuning NLLB-200, which already carries Twi: `aka_Latn` and
`twi_Latn` are both in its 202 languages, `ɛ`/`ɔ` round-trip through its
tokeniser losslessly, and words like `adwuma` are single tokens rather than
byte fallback.
"""

from .data import (
    PairFilter,
    format_source,
    load_pairs,
    split_pairs,
)

__all__ = ["PairFilter", "format_source", "load_pairs", "split_pairs"]
