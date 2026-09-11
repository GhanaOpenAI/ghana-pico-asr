"""Pair selection for the text-recovery stage."""

import pytest

from ghana_pico_asr.recovery.data import (
    HUMAN_SOURCES,
    PairFilter,
    format_source,
    pair_uer,
    split_pairs,
)


def test_units_are_joined_by_default():
    """Joined input costs 104 NLLB tokens against 246 spaced, for the same
    content — 2.4x the compute for nothing."""
    assert format_source("ɔ y ɛ n e h o") == "ɔyɛneho"
    assert format_source("ɔ y ɛ n e h o", spaced=True) == "ɔ y ɛ n e h o"


def test_pair_uer_measures_against_the_reference():
    assert pair_uer("a b c", "a b c") == 0.0
    assert pair_uer("a b c", "a b") == pytest.approx(0.5)
    assert pair_uer("", "a b c d") == pytest.approx(1.0)
    # An empty reference cannot be scored; treated as worst rather than 0/0.
    assert pair_uer("a b", "") == 1.0


def test_split_is_keyed_on_text_not_row_id():
    """The same sentence appears in more than one corpus. Keyed on id, a
    recovery model would be evaluated on sentences it had memorised."""
    rows = [
        {"id": "a1", "text": "Ɔyɛ ne ho", "units": "x"},
        {"id": "b7", "text": "Ɔyɛ ne ho", "units": "y"},   # same text, other corpus
        {"id": "c3", "text": "Mepa wo kyɛw", "units": "z"},
    ]
    out = split_pairs(rows)
    where = {}
    for name, rs in out.items():
        for r in rs:
            where.setdefault(r["text"], set()).add(name)
    # Every copy of a sentence lands in exactly one split.
    assert all(len(v) == 1 for v in where.values())
    assert sum(len(v) for v in out.values()) == 3


def test_split_proportions_are_stable():
    rows = [{"text": f"sentence number {i}", "units": "a b"} for i in range(4000)]
    out = split_pairs(rows, val_pct=2, test_pct=2)
    assert 0.01 < len(out["val"]) / 4000 < 0.035
    assert 0.01 < len(out["test"]) / 4000 < 0.035
    assert len(out["train"]) > 3700
    # Deterministic across calls.
    assert [r["text"] for r in split_pairs(rows)["val"]] == [r["text"] for r in out["val"]]


def test_human_sources_match_the_config_ids():
    """Ids 2 and 3 are the machine-transcribed corpora."""
    from ghana_pico_asr import config as C

    assert HUMAN_SOURCES == {1, 4, 5, 7}
    machine = {C.SOURCE_IDS["asr"], C.SOURCE_IDS["kuma"]}
    assert machine.isdisjoint(HUMAN_SOURCES)
    # Every registered source is accounted for as human or machine.
    assert HUMAN_SOURCES | machine | {C.SOURCE_IDS["multispk"]} == set(C.SOURCE_IDS.values())


def test_filter_defaults_are_the_measured_ones():
    f = PairFilter()
    assert f.max_uer == 0.5          # keeps ~86% of the published set
    assert f.machine_ratio == 0.5    # machine targets augment, never dominate
    assert f.min_units == 8


def test_cer_and_wer():
    from ghana_pico_asr.recovery.evaluate import cer, wer

    assert cer("abc", "abc") == 0.0
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert cer("", "abc") == pytest.approx(1.0)
    assert wer("a b c", "a b c") == 0.0
    assert wer("a b", "a b c") == pytest.approx(1 / 3)
    assert wer("", "") == 0.0


def test_evaluation_reports_the_do_nothing_baseline():
    """Joined units are already Twi-shaped, so CER alone flatters a model that
    only inserts spaces. The baseline makes that visible."""
    import inspect as _i

    from ghana_pico_asr.recovery import evaluate

    src = _i.getsource(evaluate.score_model)
    assert '"baseline_cer"' in src
    assert "exact_match" in src
