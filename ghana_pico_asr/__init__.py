"""ghana-pico-asr — a compact grapheme-unit speech recogniser.

The acoustic model maps audio to a sequence of grapheme units (for Twi, the
1-2 character graphemes of Akan orthography). It is deliberately not a full
ASR system: a separate text-recovery model turns unit sequences into words.

Languages are pluggable — see :mod:`ghana_pico_asr.languages`.
"""

__version__ = "0.1.0"
