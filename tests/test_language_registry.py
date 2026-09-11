"""The language abstraction is what lets this repo support more than Twi."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghana_pico_asr.languages import (  # noqa: E402
    DEFAULT_LANGUAGE,
    LANGUAGES,
    Language,
    get_language,
)
from ghana_pico_asr.languages.twi import TWI  # noqa: E402


def test_registry_lookup_and_aliases():
    assert get_language("twi") is TWI
    for alias in ("tw", "ak", "akan", "asante", "twi_asante", "TWI"):
        assert get_language(alias) is TWI
    assert get_language(None).code == DEFAULT_LANGUAGE


def test_unknown_language_error_lists_options_and_how_to_add():
    with pytest.raises(KeyError) as e:
        get_language("ewe")
    msg = str(e.value)
    assert "twi" in msg and "languages/" in msg


def test_inventory_ordering_is_stable():
    """Unit ids are baked into every aligned feature store, so the sorted
    inventory order must not drift or previously aligned data is mislabelled."""
    units = TWI.units
    assert units == sorted(units)
    assert len(units) == 38
    assert units[0] == "a" and "ky" in units and "ɛ" in units


def test_every_unit_is_alignable():
    for u in TWI.units:
        rom = TWI.romanize_unit(u)
        assert rom and all("a" <= c <= "z" for c in rom), u


def test_a_new_language_needs_no_model_changes():
    """A Language is data, so adding one touches no model/training code."""
    toy = Language(
        code="toy",
        name="Toy",
        multigraphs=frozenset({"ph", "sh"}),
        singles=frozenset("abcdefghijklmnopqrstuvwxyz"),
        fold={"ä": "a"},
        romanize={},
        contrasts=(("a", "b"),),
    )
    assert toy.max_unit_len == 2
    assert toy.segment_word("shophat") == ["sh", "o", "ph", "a", "t"]
    assert toy.normalize("ShÄp!! 42 ph") == "shap ph"
    assert toy.segment("ShÄp") == [["sh", "a", "p"]]
    # longest-match generalises beyond 2 characters
    tri = Language(
        code="t3", name="T3", multigraphs=frozenset({"sch", "ch"}),
        singles=frozenset("abcdefghijklmnopqrstuvwxyz"), fold={}, romanize={},
    )
    assert tri.max_unit_len == 3
    assert tri.segment_word("schach") == ["sch", "a", "ch"]


def test_contrasts_are_language_data():
    assert ("ɛ", "e") in TWI.contrasts and ("ɔ", "o") in TWI.contrasts
    assert all(len(p) == 2 for p in TWI.contrasts)
    # the pairs must be real units, or the contrast report silently reports nothing
    for a, b in TWI.contrasts:
        assert a in TWI.units and b in TWI.units


def test_alphabet_excludes_folded_characters():
    assert "ŋ" not in TWI.alphabet and "ŋ" in TWI.fold
    assert "ɛ" in TWI.alphabet and "ɛ" not in TWI.fold
    assert LANGUAGES[TWI.code] is TWI
