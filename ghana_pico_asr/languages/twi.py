"""Twi (Asante Twi / Akan) grapheme-unit inventory.

Akan orthography spells a number of single phonemes with two letters, so the
unit inventory is the single letters plus those digraphs::

    "nkyekyɛmu"  ->  n | ky | e | ky | ɛ | m | u
    "ahwɛ"       ->  a | hw | ɛ
    "adwuma"     ->  a | dw | u | m | a

Two alphabets are in play. Label units keep the Twi orthography, so ``ɛ``/``e``
and ``ɔ``/``o`` stay distinct classes. Aligner tokens are romanised to ASCII
because the MMS CTC vocabulary has no ``ɛ`` or ``ɔ`` — those pairs collapse for
the aligner, which costs nothing since it only supplies timings, and the model
still learns the contrast from audio.
"""

from __future__ import annotations

from .base import Language

#: Two-letter graphemes spelling a single Twi phoneme. Nasal + stop clusters
#: (nk, nt, mp, ns, ...) are deliberately absent: they are two phonemes.
DIGRAPHS = frozenset(
    {
        "ky",  # /tɕ/
        "gy",  # /dʑ/
        "hy",  # /ɕ/
        "ny",  # /ɲ/
        "tw",  # /tɕʷ/
        "dw",  # /dʑʷ/
        "kw",  # /kʷ/
        "gw",  # /gʷ/
        "hw",  # /ɕʷ/
        "nw",  # /ŋʷ/
    }
)

VOWELS = frozenset("aeɛioɔu")
CONSONANTS = frozenset("bdfghklmnprstwy")
#: Only reached via English code-switching, which is common in Ghanaian speech.
FOREIGN = frozenset("cjqvxz")

#: Letters outside the label alphabet, folded during normalisation. ``ɛ`` and
#: ``ɔ`` are deliberately absent — they are real classes and must survive.
#: Pleasant side effect: "ŋw" folds to "nw", the digraph it spells.
FOLD = {
    "ŋ": "n", "ɲ": "n", "ɩ": "i", "ɪ": "i", "ʊ": "u", "ə": "e", "ɐ": "a",
    "ʃ": "s", "ʒ": "z", "ɡ": "g", "æ": "a", "ø": "o", "œ": "o", "ß": "s",
    "ð": "d", "þ": "t", "đ": "d", "ħ": "h", "ł": "l",
}

#: The first two pairs romanise identically, so the CTC aligner cannot tell
#: them apart — the model can only learn them from audio (or from ATR vowel
#: harmony context). They are the honest test of acoustic learning.
CONTRASTS = (
    ("ɛ", "e"),
    ("ɔ", "o"),
    ("ky", "gy"),
    ("tw", "dw"),
    ("hy", "hw"),
    ("ny", "nw"),
    ("k", "kw"),
)

TWI = Language(
    code="twi",
    name="Twi (Asante)",
    multigraphs=DIGRAPHS,
    singles=VOWELS | CONSONANTS | FOREIGN,
    fold=FOLD,
    romanize={"ɛ": "e", "ɔ": "o"},
    contrasts=CONTRASTS,
)
