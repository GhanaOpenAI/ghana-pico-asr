"""Training-pair construction for the text-recovery stage."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghana_pico_asr.pairs import (  # noqa: E402
    Pair,
    describe_mix,
    mix,
    read_pairs,
    write_pairs,
)


def P(i, origin="real"):
    return Pair(units=f"u{i}", text=f"t{i}", origin=origin, id=str(i))


REAL = [P(i) for i in range(100)]


def test_even_mix_when_real_pairs_are_plentiful():
    syn = [P(i, "synthetic") for i in range(40)]
    mixed, r = mix(syn, REAL, ratio=1.0)
    assert r["real_used"] == 40
    assert r["achieved_ratio"] == 1.0
    assert len(mixed) == 80
    assert not r["real_capped_by_availability"]


def test_all_real_pairs_used_when_new_domain_is_larger():
    """The specified rule: if the new domain outnumbers the real pairs, use
    every real pair rather than sampling with replacement — repeating the same
    examples over-weights them without adding information."""
    syn = [P(i, "synthetic") for i in range(500)]
    mixed, r = mix(syn, REAL, ratio=1.0)
    assert r["real_used"] == len(REAL) == 100
    assert r["real_requested"] == 500
    assert r["real_capped_by_availability"] is True
    assert len(mixed) == 600
    # and the achieved ratio is reported honestly, not as the requested 1.0
    assert r["achieved_ratio"] == pytest.approx(0.2)


def test_real_pairs_are_never_duplicated():
    syn = [P(i, "synthetic") for i in range(500)]
    mixed, _ = mix(syn, REAL, ratio=1.0)
    ids = [p.id for p in mixed if p.origin == "real"]
    assert len(ids) == len(set(ids))


def test_ratio_is_tunable_for_balance_experiments():
    syn = [P(i, "synthetic") for i in range(40)]
    for ratio, expect in ((0.0, 0), (0.5, 20), (1.0, 40), (2.0, 80)):
        _, r = mix(syn, REAL, ratio=ratio)
        assert r["real_used"] == expect, ratio
    with pytest.raises(ValueError, match="ratio must be >= 0"):
        mix(syn, REAL, ratio=-1.0)


def test_mix_is_reproducible_under_a_seed():
    syn = [P(i, "synthetic") for i in range(60)]
    a, _ = mix(syn, REAL, seed=7)
    b, _ = mix(syn, REAL, seed=7)
    c, _ = mix(syn, REAL, seed=8)
    key = lambda ps: [(p.id, p.origin) for p in ps]  # noqa: E731
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_no_shuffle_keeps_synthetic_first():
    syn = [P(i, "synthetic") for i in range(5)]
    mixed, _ = mix(syn, REAL, ratio=1.0, shuffle=False)
    assert [p.origin for p in mixed[:5]] == ["synthetic"] * 5


def test_describe_mix_warns_when_capped():
    syn = [P(i, "synthetic") for i in range(500)]
    _, r = mix(syn, REAL, ratio=1.0)
    text = describe_mix(r)
    assert "only 100" in text and "make-pairs" in text
    _, r2 = mix([P(i, "synthetic") for i in range(10)], REAL, ratio=1.0)
    assert "only" not in describe_mix(r2)


def test_pairs_round_trip_jsonl(tmp_path):
    path = tmp_path / "p.jsonl"
    src = [
        Pair(units="a b", text="ab", origin="real", source="ds", id="1", reference_units="a b c"),
        Pair(units="c d", text="cd", origin="synthetic", source="txt", id="2"),
    ]
    write_pairs(src, str(path))
    back = read_pairs(str(path))
    assert [p.units for p in back] == ["a b", "c d"]
    assert back[0].reference_units == "a b c"
    # synthetic pairs omit reference_units rather than writing null
    line2 = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert "reference_units" not in line2


def test_read_pairs_rejects_files_missing_required_keys(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"units": "a b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="needs 'units' and 'text'"):
        read_pairs(str(p))
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_pairs(str(p))


def test_real_pair_units_differ_from_reference():
    """The whole point of real pairs: their unit side is lossy. If units and
    reference_units were identical there would be nothing for the recovery
    model to learn to repair."""
    p = Pair(
        units="hy ɛ k r t o f o",
        text="hyɛɛ Kristofo",
        origin="real",
        reference_units="hy ɛ ɛ k r i s t o f o",
    )
    assert p.units != p.reference_units
    assert len(p.units.split()) < len(p.reference_units.split())


# ------------------------------------------------- stratified replay sampling


def imbalanced_corpus():
    """Mirrors the real corpus: film dialogue ~75%, health ~20%, TTS ~5%."""
    return (
        [Pair(units=f"k{i}", text=f"t{i}", origin="real", source="kuma", id=f"k{i}")
         for i in range(7500)]
        + [Pair(units=f"a{i}", text=f"t{i}", origin="real", source="asr", id=f"a{i}")
           for i in range(2000)]
        + [Pair(units=f"s{i}", text=f"t{i}", origin="real", source="tts", id=f"s{i}")
           for i in range(500)]
    )


def test_proportional_replay_mirrors_the_corpus_imbalance():
    real = imbalanced_corpus()
    syn = [P(i, "synthetic") for i in range(3000)]
    _, r = mix(syn, real, ratio=1.0, balance="proportional", seed=1)
    by = r["replay_by_source"]
    assert sum(by.values()) == 3000
    # 75/20/5 of 3000
    assert by["kuma"] == pytest.approx(2250, abs=5)
    assert by["asr"] == pytest.approx(600, abs=5)
    assert by["tts"] == pytest.approx(150, abs=5)


def test_equal_replay_stops_one_domain_dominating():
    real = imbalanced_corpus()
    syn = [P(i, "synthetic") for i in range(3000)]
    _, r = mix(syn, real, ratio=1.0, balance="equal", seed=1)
    by = r["replay_by_source"]
    assert sum(by.values()) == 3000, "the requested total must still be honoured"
    # tts only has 500, so it is capped and the shortfall is redistributed
    assert by["tts"] == 500
    assert by["kuma"] > by["tts"] and by["asr"] > by["tts"]
    # but far less lopsided than proportional
    assert by["kuma"] / by["asr"] < 2.0


def test_replay_report_states_what_is_available():
    real = imbalanced_corpus()
    _, r = mix([P(i, "synthetic") for i in range(100)], real)
    assert r["available_by_source"] == {"kuma": 7500, "asr": 2000, "tts": 500}


def test_using_all_real_pairs_skips_stratification():
    real = imbalanced_corpus()
    syn = [P(i, "synthetic") for i in range(50_000)]  # far more than available
    _, r = mix(syn, real, ratio=1.0)
    assert r["real_used"] == 10_000
    assert r["replay_balance"] == "all-used"
    assert sum(r["replay_by_source"].values()) == 10_000


def test_invalid_balance_is_rejected():
    real = imbalanced_corpus()
    with pytest.raises(ValueError, match="proportional.*equal"):
        mix([P(i, "synthetic") for i in range(10)], real, balance="nonsense")


def test_stratified_replay_never_duplicates():
    real = imbalanced_corpus()
    syn = [P(i, "synthetic") for i in range(5000)]
    mixed, _ = mix(syn, real, ratio=1.0, balance="equal", seed=3)
    ids = [p.id for p in mixed if p.origin == "real"]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------- pair source loading


def test_load_real_pairs_prefers_an_explicit_file(tmp_path):
    from ghana_pico_asr.pairs import load_real_pairs, write_csv

    path = tmp_path / "mine.csv"
    write_csv([Pair(units="a b", text="ab", origin="real", source="x", id="1")], str(path))
    pairs, where = load_real_pairs(str(path))
    assert len(pairs) == 1 and where == str(path)


def test_load_real_pairs_rejects_a_bad_reference():
    from ghana_pico_asr.pairs import load_real_pairs

    with pytest.raises(SystemExit, match="neither an existing file nor an HF dataset"):
        load_real_pairs("not_a_path_or_repo")


def test_csv_and_jsonl_are_both_accepted(tmp_path):
    from ghana_pico_asr.pairs import read_pairs, write_csv, write_pairs

    src = [Pair(units="a b", text="ab", origin="real", source="s", id="1",
                reference_units="a b c")]
    csv_p, jsonl_p = tmp_path / "p.csv", tmp_path / "p.jsonl"
    write_csv(src, str(csv_p))
    write_pairs(src, str(jsonl_p))
    a, b = read_pairs(str(csv_p)), read_pairs(str(jsonl_p))
    assert a[0].units == b[0].units == "a b"
    assert a[0].reference_units == b[0].reference_units == "a b c"


def test_pairs_csv_missing_columns_is_rejected(tmp_path):
    from ghana_pico_asr.pairs import read_pairs

    p = tmp_path / "bad.csv"
    p.write_text("units,notext\na b,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs \\['text'\\] column"):
        read_pairs(str(p))


def test_hf_pairs_repo_is_declared_per_language():
    from ghana_pico_asr.pairs import HF_PAIRS_REPO

    assert "twi" in HF_PAIRS_REPO and "/" in HF_PAIRS_REPO["twi"]


# ----------------------------------------------------------------- source ids


#: The ids as first published. Adding a source is fine; renumbering one of
#: these would mislabel pair data already released, so they are frozen here.
PUBLISHED_SOURCE_IDS = {"tts": 1, "asr": 2, "kuma": 3, "female": 4,
                        "agric": 5, "multispk": 6, "bible": 7}


def test_source_ids_are_append_only():
    """Published pairs label their corpus with a number, so the dataset does
    not hard-code corpus identities. New sources may be appended; an existing
    id must never be reused for a different corpus."""
    from ghana_pico_asr import config as C

    for name, sid in PUBLISHED_SOURCE_IDS.items():
        assert C.SOURCE_IDS.get(name) == sid, (
            f"{name!r} was published as id {sid}; changing it mislabels "
            "pair data already released"
        )
    ids = list(C.SOURCE_IDS.values())
    assert len(ids) == len(set(ids)), "two sources share an id"
    assert min(ids) == 1 and max(ids) == len(ids), "ids must be a dense 1..n range"


def test_reverse_mapping_is_consistent():
    from ghana_pico_asr import config as C

    assert C.SOURCE_NAMES == {v: k for k, v in C.SOURCE_IDS.items()}
    assert C.source_id("kuma") == 3


def test_unknown_source_says_which_id_is_free():
    from ghana_pico_asr import config as C

    nxt = max(C.SOURCE_IDS.values()) + 1
    with pytest.raises(KeyError, match=f"next free id is {nxt}"):
        C.source_id("ewe")


def test_stratification_works_on_numeric_source_labels():
    """The replay mix groups by whatever is in `source`, so it must behave the
    same with numeric ids as with names."""
    real = (
        [Pair(units=f"a{i}", text="t", origin="real", source="3", id=f"c{i}")
         for i in range(7500)]
        + [Pair(units=f"b{i}", text="t", origin="real", source="2", id=f"b{i}")
           for i in range(2000)]
        + [Pair(units=f"c{i}", text="t", origin="real", source="1", id=f"a{i}")
           for i in range(500)]
    )
    syn = [P(i, "synthetic") for i in range(3000)]
    _, r = mix(syn, real, ratio=1.0, balance="proportional", seed=1)
    by = r["replay_by_source"]
    assert set(by) == {"1", "2", "3"}
    assert by["3"] > by["2"] > by["1"]
    assert sum(by.values()) == 3000
