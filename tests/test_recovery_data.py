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


def test_uer_column_is_cached_between_runs(tmp_path):
    """Scoring 282k pairs takes ~12 min and gives the same answer every time.

    A smoke run that keeps 4k pairs would otherwise pay it in full.
    """
    from ghana_pico_asr.recovery.data import _uer_column

    rows = [{"units": "a b c", "reference_units": "a b d"} for _ in range(5)]
    first = _uer_column(rows, str(tmp_path), "k")
    assert first == [pytest.approx(1 / 3)] * 5
    assert (tmp_path / "uer_k.json").exists()

    # A cache of the wrong length is ignored rather than trusted.
    stale = [{"units": "x", "reference_units": "x"}] * 3
    assert len(_uer_column(stale, str(tmp_path), "k")) == 3

    # No cache dir still works.
    assert _uer_column(rows, None, "k") == first


def test_safetensors_is_required_for_the_base_model():
    """transformers 5.x refuses torch.load on torch < 2.6 (CVE-2025-32434),
    and NLLB ships both a .bin and safetensors."""
    import io as _io

    src = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    assert "use_safetensors=True" in src


def test_training_arguments_are_valid_for_the_pinned_transformers():
    """Every kwarg the job passes must exist in the pinned transformers.

    `transformers>=4.44` floated to a release that had dropped `warmup_ratio`,
    failing a run that had worked hours earlier. Pins fix the drift; this
    catches a bad kwarg before a job is scheduled.
    """
    import inspect
    import io as _io
    import re

    from transformers import Seq2SeqTrainingArguments

    src = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    block = src[src.index("targs = Seq2SeqTrainingArguments("):src.index("trainer = Seq2SeqTrainer(")]
    used = re.findall(r"^\s{8}(\w+)=", block, re.M)
    sig = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    assert used, "no kwargs found — the parser needs updating"
    assert [k for k in used if k not in sig] == []


def test_trainer_supports_both_model_families():
    """NLLB conditions on language codes; the T5 family has none.

    Vanilla t5-base is excluded on purpose: its tokeniser drops ɛ and ɔ
    entirely (`Ɔyɛ ne ho` -> `y ne ho`), destroying the phonemic contrasts the
    grapheme-unit inventory exists to carry. mT5 and ByT5 round-trip losslessly
    and need no vocabulary surgery.
    """
    import io as _io

    src = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    # Family detected from the tokeniser, not hardcoded from the model name.
    assert "is_nllb = args.lang_code in probe.get_vocab()" in src
    # T5 gets a task prefix in place of language conditioning.
    assert "restore twi: " in src
    # A forced BOS is NLLB-only.
    assert "forced_bos = tok.convert_tokens_to_ids(args.lang_code) if is_nllb else None" in src
    # LoRA targets the right layer names per family.
    assert '"wi_0", "wi_1", "wo"' in src


def test_scorer_makes_the_language_token_optional():
    import inspect

    from ghana_pico_asr.recovery.evaluate import score_model

    assert inspect.signature(score_model).parameters["lang_code"].annotation == "str | None"


def test_baseline_excludes_the_task_prefix():
    """The do-nothing baseline must score units alone.

    Counting `restore twi: ` as errors inflated the T5 family's baseline to
    0.588 against NLLB's 0.391 on identical data, making the families
    incomparable on the one number the comparison turns on.
    """
    import io as _io

    from ghana_pico_asr.recovery.evaluate import score_model

    src = _io.open("ghana_pico_asr/recovery/evaluate.py", encoding="utf-8").read()
    assert 'r.get("raw_source") or r["source_text"]' in src
    job = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    assert 'r["raw_source"] = r["source_text"]' in job


def test_clean_pairs_are_off_by_default():
    """At inference the model only ever sees noisy units. Training heavily on
    clean input teaches it to trust what it is given — a better formatter and
    a worse corrector — so this is opt-in."""
    from ghana_pico_asr.recovery.data import PairFilter

    assert PairFilter().clean_ratio == 0.0


def test_clean_pairs_use_reference_units_and_are_marked():
    """A clean pair teaches restoration alone: the reference units are the
    target text with boundaries, capitalisation and punctuation stripped, so
    there are no errors to correct — only the mapping to learn."""
    import io as _io

    src = _io.open("ghana_pico_asr/recovery/data.py", encoding="utf-8").read()
    assert 'format_source(r["reference_units"]' in src
    assert 'c["origin"] = "clean-ref"' in src
    # Real pairs stay labelled, so a run can report what it actually trained on.
    assert 'r["origin"] = "real"' in src
    # The count is reported, not silent.
    assert 'report["clean_ref_added"]' in src


def test_clean_augmentation_preserves_the_target():
    """Only the source changes; the text being recovered must be identical, or
    the two pair types teach different tasks."""
    import io as _io

    src = _io.open("ghana_pico_asr/recovery/data.py", encoding="utf-8").read()
    block = src[src.index("if flt.clean_ratio > 0:"):src.index('report["final"]')]
    assert "c = dict(r)" in block          # inherits target_text unchanged
    assert 'c["target_text"]' not in block  # and never overwrites it


def test_a_completed_evaluation_survives_a_failed_metrics_write():
    """The bucket is object storage: an empty directory does not reliably
    persist between creation and a later open(). A finished evaluation must
    not be lost to that — it cost GPU time to produce."""
    import io as _io

    src = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    block = src[src.index('metrics.json'):]
    assert "except OSError" in src[src.index("os.makedirs(out_dir, exist_ok=True)\n        with open"):]
    # And the numbers are printed before any write is attempted.
    assert src.index('print(f"[test] ') < src.index('"metrics.json"), "w"')


def test_tokenizer_extension_trains_the_new_embeddings():
    """Adding tokens without training their embeddings is worse than useless.

    t5 emits UNK for Ɔ Ɛ ɔ ɛ and round-trips "Ɔyɛ ne ho" to "y ne ho". Adding
    the four characters resizes the embedding matrix, but LoRA does not touch
    embeddings — so without `modules_to_save` the new rows stay at their random
    initialisation for the entire run and the model can never read or write
    those vowels.
    """
    import io as _io

    src = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    assert "resize_token_embeddings(len(tok))" in src
    assert 'modules_to_save=(["shared", "lm_head"] if added else None)' in src
    # Resize must happen before the model is wrapped for LoRA.
    assert src.index("resize_token_embeddings") < src.index("get_peft_model")


def test_the_two_clean_sources_are_independent():
    """They teach different things and are meant to combine.

    Same-corpus clean pairs give the model the utterance it will meet
    corrupted, paired with the units it should have had — fixing the mapping on
    exactly the material the recogniser gets wrong. Outside text is the only
    source of sentences it has never been asked to produce, which is what
    failed to transfer to Waxal. Picking one or the other, as an earlier
    version did, gets only half the job.
    """
    import io as _io

    from ghana_pico_asr.recovery.data import PairFilter

    f = PairFilter(clean_ratio=0.5, clean_text_ratio=2.0)
    assert f.clean_ratio == 0.5 and f.clean_text_ratio == 2.0

    src = _io.open("ghana_pico_asr/recovery/data.py", encoding="utf-8").read()
    body = src[src.index("    extra: list[dict] = []"):src.index('    report["final"]')]
    # Two independent ifs, not if/elif — both can contribute to one run.
    assert body.count("if flt.clean") == 2
    assert "elif" not in body
    assert '"clean-ref"' in body and '"clean-text"' in body
    # The realised mix is reported, so a run records what it actually trained on.
    assert 'report["mix"]' in body


def test_clean_text_source_is_a_different_register():
    """Same-corpus clean pairs cannot fix a domain failure: their text
    distribution is the one that failed to transfer."""
    from ghana_pico_asr.recovery.data import CLEAN_TEXT_REPO, PairFilter

    assert CLEAN_TEXT_REPO == "ghananlpcommunity/pristine-twi-english-parallel-sentences"
    # Defaults to the external corpus, not reference_units.
    assert PairFilter().clean_text_repo == CLEAN_TEXT_REPO
    assert PairFilter().clean_ratio == 0.0   # still opt-in


def test_clean_pairs_never_reach_val_or_test():
    """Their input is already correct, so they are free wins that deflate both
    CER and the baseline. Mixing 2:1 clean moved the reported baseline from
    0.4395 to 0.2876 with nothing about the task changed."""
    from ghana_pico_asr.recovery.data import split_pairs

    rows = [{"text": f"sentence {i}", "origin": "real"} for i in range(500)]
    rows += [{"text": f"clean {i}", "origin": "clean-text"} for i in range(500)]
    rows += [{"text": f"cref {i}", "origin": "clean-ref"} for i in range(200)]
    out = split_pairs(rows)

    for split in ("val", "test"):
        assert all(r["origin"] == "real" for r in out[split]), split
    assert sum(1 for r in out["train"] if r["origin"] != "real") == 700
    # Rows with no origin at all still split normally.
    assert split_pairs([{"text": f"x{i}"} for i in range(400)])["val"]


def test_sentencepiece_extension_has_a_spacing_ceiling():
    """Adding ɛ/ɔ to a SentencePiece tokeniser breaks word spacing.

    An added token carries word-boundary semantics, so the tokeniser cannot
    round-trip its own input: "Ɔyɛ ne ho" becomes "Ɔ yɛ ne ho" and "yɛhunu"
    becomes "yɛ hunu". Every Twi word with ɛ or ɔ mid-word gains a space, and
    no amount of training can recover it. Recorded so nobody re-runs the
    experiment expecting a different answer.
    """
    pytest.importorskip("sentencepiece")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google-t5/t5-base")
    tok.add_tokens(["Ɔ", "Ɛ", "ɔ", "ɛ"])
    rt = lambda s: tok.decode(  # noqa: E731
        tok(s, add_special_tokens=False)["input_ids"], skip_special_tokens=True
    ).strip()

    assert rt("Ɔyɛ ne ho") == "Ɔ yɛ ne ho"   # mid-word ɛ/ɔ gains a space
    assert rt("yɛhunu") == "yɛ hunu"
    assert rt("Amanneɛ") == "Amanneɛ"        # word-final is unaffected

    # The models we prefer have no such ceiling.
    for name in ("Qwen/Qwen2.5-0.5B", "google/gemma-3-270m"):
        t = AutoTokenizer.from_pretrained(name)
        assert t.decode(t("Ɔyɛ ne ho")["input_ids"],
                        skip_special_tokens=True).strip() == "Ɔyɛ ne ho", name


def test_eval_only_resizes_the_base_model_too():
    """An adapter trained with added tokens carries a resized embedding matrix.

    `--eval-only` builds a fresh base model, so it must apply the same resize
    before loading — otherwise the shapes differ by the added tokens plus the
    original matrix's padding (t5-base: 32,104 against 32,128) and the load
    fails.
    """
    import io as _io

    src = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    branch = src[src.index("if args.eval_only:"):src.index("trainer.model = model")]
    assert "base.resize_token_embeddings(len(tok))" in branch
    # And before the adapter is attached, not after.
    assert branch.index("resize_token_embeddings") < branch.index("PeftModel.from_pretrained")
