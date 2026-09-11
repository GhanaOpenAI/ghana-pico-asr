"""What a language must provide to be supported.

The acoustic model is language-agnostic: it maps audio frames to units. What
differs per language is the *unit inventory* and how text is split into it.
A `Language` supplies that, so adding a language means adding one module here
rather than touching the model, dataset, or training code.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    """A grapheme-unit inventory and the rules for segmenting text into it."""

    #: BCP-47-ish short code used on the command line and in checkpoints.
    code: str
    #: Human-readable name.
    name: str
    #: Multi-character graphemes that spell a single phoneme, longest-first
    #: matching. For Twi these are the digraphs (ky, gy, hw, ...).
    multigraphs: frozenset[str]
    #: Single characters that are units in their own right.
    singles: frozenset[str]
    #: Characters folded to an inventory character during normalisation.
    #: Anything not in `alphabet` after folding is stripped.
    fold: dict[str, str]
    #: Characters mapped to the CTC aligner's ASCII vocabulary. Units whose
    #: romanisation is empty are unalignable and dropped.
    romanize: dict[str, str]
    #: Apostrophe-like characters deleted rather than tokenised (they mark
    #: elision in Twi and carry no acoustics).
    apostrophes: str = "'’‘ʼ`´"
    #: Unit pairs worth reporting separately during evaluation — contrasts the
    #: aligner cannot represent, or that are otherwise easy to confuse.
    contrasts: tuple[tuple[str, str], ...] = ()

    # -- derived -------------------------------------------------------- #

    @property
    def alphabet(self) -> frozenset[str]:
        """The only characters that survive normalisation."""
        return self.singles | frozenset("".join(self.multigraphs))

    @property
    def units(self) -> list[str]:
        """Full a-priori inventory, sorted. The live vocab is a subset."""
        return sorted(u for u in (self.singles | self.multigraphs) if self.romanize_unit(u))

    @property
    def max_unit_len(self) -> int:
        return max((len(u) for u in self.multigraphs), default=1)

    # -- text processing ------------------------------------------------- #

    def _strip_accents(self, text: str) -> str:
        d = unicodedata.normalize("NFD", text)
        kept = "".join(c for c in d if not unicodedata.combining(c))
        return unicodedata.normalize("NFC", kept)

    def normalize(self, text: str) -> str:
        """Lower-case, fold, strip punctuation/digits, collapse whitespace.

        Guarantees the result contains only `alphabet` characters and single
        spaces, so `segment_word` can never produce an out-of-inventory unit.
        """
        text = self._strip_accents(text).lower().translate(self._fold_table)
        text = self._apostrophe_re.sub("", text)
        # A word containing a digit is dropped whole: there is no way to know
        # how a speaker read "2024", so its audio must not be labelled.
        words = [w for w in text.split() if not self._digit_re.search(w)]
        text = self._non_alpha_re.sub(" ", " ".join(words))
        return " ".join(text.split())

    def segment_word(self, word: str) -> list[str]:
        """Greedy longest-match a normalised word into units."""
        out: list[str] = []
        i, n, mx = 0, len(word), self.max_unit_len
        while i < n:
            for size in range(min(mx, n - i), 1, -1):
                if word[i : i + size] in self.multigraphs:
                    out.append(word[i : i + size])
                    i += size
                    break
            else:
                out.append(word[i])
                i += 1
        return out

    def segment(self, text: str) -> list[list[str]]:
        """Normalise and return one list of units per word.

        Every returned unit is in `units` and has a non-empty romanisation, so
        callers can index a unit-id table without guarding.
        """
        valid = self._valid_units
        words: list[list[str]] = []
        for word in self.normalize(text).split():
            units = [u for u in self.segment_word(word) if u in valid]
            if units:
                words.append(units)
        return words

    def romanize_unit(self, unit: str) -> str:
        """Map a unit onto the aligner's ASCII alphabet ("" if unusable)."""
        out = "".join(self.romanize.get(c, c) for c in unit)
        return "".join(c for c in out if "a" <= c <= "z")

    def aligner_token(self, unit: str) -> str:
        """Space-separated characters, as the MMS tokenizer expects.

        `get_spans` splits each token on " " and asserts the CTC path visits
        exactly those characters, so a digraph must become "k y".
        """
        return " ".join(self.romanize_unit(unit))

    # -- lazily built lookups ------------------------------------------- #

    def __post_init__(self):
        object.__setattr__(self, "_fold_table", str.maketrans(self.fold))
        object.__setattr__(self, "_digit_re", re.compile(r"\d"))
        object.__setattr__(
            self, "_apostrophe_re", re.compile(f"[{re.escape(self.apostrophes)}]")
        )
        object.__setattr__(
            self,
            "_non_alpha_re",
            re.compile(f"[^{re.escape(''.join(sorted(self.alphabet)))}]+"),
        )
        object.__setattr__(
            self, "_valid_units", frozenset(u for u in self.units if self.romanize_unit(u))
        )


def flatten(words: list[list[str]]) -> list[str]:
    return [u for w in words for u in w]


def build_tokens_starred(lang: Language, words: list[list[str]]) -> tuple[list[str], list[str]]:
    """Build the aligner's paired ``(tokens_starred, text_starred)`` lists.

    ``<star>`` is a wildcard that can absorb arbitrary audio. One goes at each
    end (leading/trailing silence) and one at every word boundary (pauses,
    breaths) — but never between units inside a word, which would let the
    aligner smear phoneme boundaries apart.
    """
    tokens: list[str] = ["<star>"]
    text: list[str] = ["<star>"]
    for i, units in enumerate(words):
        if i:
            tokens.append("<star>")
            text.append("<star>")
        for u in units:
            tokens.append(lang.aligner_token(u))
            text.append(u)
    tokens.append("<star>")
    text.append("<star>")
    return tokens, text
