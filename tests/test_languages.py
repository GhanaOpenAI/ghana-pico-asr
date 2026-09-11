"""Unit tests for Twi 2-char grapheme segmentation and aligner token building."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghana_pico_asr.languages import build_tokens_starred, flatten, get_language  # noqa: E402

G = get_language("twi")


def test_digraph_segmentation():
    assert G.segment_word("nkyekyɛmu") == ["n", "ky", "e", "ky", "ɛ", "m", "u"]
    assert G.segment_word("ahwɛ") == ["a", "hw", "ɛ"]
    assert G.segment_word("gyae") == ["gy", "a", "e"]
    assert G.segment_word("adwuma") == ["a", "dw", "u", "m", "a"]
    assert G.segment_word("nyinaa") == ["ny", "i", "n", "a", "a"]
    assert G.segment_word("kwan") == ["kw", "a", "n"]


def test_nasal_clusters_stay_split():
    # nk / nt / mp / ns are two phonemes, not digraphs.
    assert G.segment_word("nkuran") == ["n", "k", "u", "r", "a", "n"]
    assert G.segment_word("nti") == ["n", "t", "i"]
    assert G.segment_word("mpo") == ["m", "p", "o"]


def test_apostrophes_are_deleted_not_tokenised():
    assert G.segment("w'aba") == [["w", "a", "b", "a"]]
    assert G.segment("y'asɔne") == [["y", "a", "s", "ɔ", "n", "e"]]


def test_digits_drop_the_whole_word():
    assert G.segment("me 2024 ba") == [["m", "e"], ["b", "a"]]


def test_punctuation_and_case():
    assert G.segment("Ɔyɛ, ne ho!") == [["ɔ", "y", "ɛ"], ["n", "e"], ["h", "o"]]


def test_romanization_collapses_open_vowels_but_labels_do_not():
    assert G.romanize_unit("ɛ") == "e"
    assert G.romanize_unit("ɔ") == "o"
    assert G.romanize_unit("ky") == "ky"
    # The label alphabet keeps them apart, which is the whole point.
    assert "ɛ" in G.units and "e" in G.units


def test_aligner_tokens_are_space_separated_chars():
    # get_spans() splits each token on " " and asserts the CTC path matches.
    assert G.aligner_token("ky") == "k y"
    assert G.aligner_token("ɛ") == "e"


def test_stars_at_edges_and_word_boundaries_only():
    words = G.segment("hwɛ me")
    tokens, text = build_tokens_starred(G, words)
    assert text == ["<star>", "hw", "ɛ", "<star>", "m", "e", "<star>"]
    assert tokens == ["<star>", "h w", "e", "<star>", "m", "e", "<star>"]
    # Never a star between units inside a word.
    assert tokens[1:3] == ["h w", "e"]


def test_every_unit_romanizes_into_ascii():
    for unit in G.units:
        rom = G.romanize_unit(unit)
        assert rom and all("a" <= c <= "z" for c in rom), unit


def test_segment_text_drops_unusable_units():
    # A bare apostrophe word disappears entirely rather than yielding [].
    assert G.segment("' ' me") == [["m", "e"]]


def test_normalize_only_emits_the_label_alphabet():
    import random

    from ghana_pico_asr.languages.twi import TWI

    # Characters plausibly present in Twi/Akan text plus assorted noise.
    pool = (
        "abcdefghijklmnopqrstuvwxyzɛɔŋɩɪʊəæøœßðþɲʃʒɡ"
        "ÁÀÂÄÉÈÊÍÌÎÓÒÔÚÙÛÑÇ0123456789 .,!?;:'’\"()[]-–—/\\@#$%&*+=<>~`|\t\n"
        "ΑΒΓдже漢字"
    )
    rng = random.Random(1234)
    for _ in range(2000):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
        norm = G.normalize(text)
        assert set(norm) <= TWI.alphabet | {" "}, repr(norm)
        assert "  " not in norm
        assert norm == norm.strip()


def test_segment_text_never_emits_an_unknown_unit():
    """The guard that stage 1's unit-id lookup depends on.

    A leaked 'ŋ' used to reach ``UNIT_TO_ID`` and kill an entire shard.
    """
    import random

    from ghana_pico_asr.prepare import UNIT_TO_ID

    pool = "abcdefghijklmnopqrstuvwxyzɛɔŋɩɪʊəæøœßðþɲʃʒɡ0123456789 '’.,-ÁÈÔÑ漢"
    rng = random.Random(99)
    inventory = set(G.units)
    for _ in range(3000):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
        for unit in flatten(G.segment(text)):
            assert unit in inventory, f"{unit!r} not in inventory"
            assert unit in UNIT_TO_ID, f"{unit!r} would KeyError in stage 1"
            assert G.romanize_unit(unit), f"{unit!r} has no aligner token"


def test_folded_letters_reach_the_right_unit():
    assert G.segment_word(G.normalize("ŋkyeŋ")) == ["n", "ky", "e", "n"]
    # "ŋw" spells the labialised velar nasal, which is the "nw" digraph.
    assert G.segment("ŋwene") == [["nw", "e", "n", "e"]]
    # An unrecognised letter (ɴ) is stripped rather than guessed at, which
    # splits the word — the aligner then gets a <star> at that boundary.
    assert G.segment("ɪɴʊə") == [["i"], ["u", "e"]]
    assert G.segment("ɪʊə") == [["i", "u", "e"]]


def test_open_vowels_are_never_folded():
    assert G.segment("ɔyɛ") == [["ɔ", "y", "ɛ"]]
    assert "ɛ" not in G.fold and "ɔ" not in G.fold
