"""Decoder-only formatting for the recovery stage."""

from ghana_pico_asr.recovery.causal import (
    IGNORE,
    build_example,
    collate,
    strip_prompt,
)


class FakeTok:
    """Character-level stand-in; ids are code points."""

    eos_token_id = 99

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_prompt_is_masked_out_of_the_loss():
    """Without masking, most of the loss is spent predicting the grapheme
    units — the one thing the pipeline already has."""
    ex = build_example(FakeTok(), "abc", "xy")
    n_ignored = sum(1 for l in ex["labels"] if l == IGNORE)
    assert n_ignored > 0
    # Every non-ignored label is part of the target (plus EOS).
    kept = [l for l in ex["labels"] if l != IGNORE]
    assert kept[-1] == FakeTok.eos_token_id
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_completion_ends_with_eos():
    """No EOS means no signal for where to stop, and generation runs to the
    token limit."""
    ex = build_example(FakeTok(), "a", "b")
    assert ex["input_ids"][-1] == FakeTok.eos_token_id
    assert ex["labels"][-1] == FakeTok.eos_token_id


def test_collate_pads_labels_with_ignore_not_pad():
    """Padding labels with the pad id trains the model to emit padding."""
    feats = [build_example(FakeTok(), "abc", "xy"), build_example(FakeTok(), "a", "z")]
    batch = collate(feats, pad_token_id=0)
    assert batch["input_ids"].shape == batch["labels"].shape
    short = batch["labels"][1].tolist()
    assert short[-1] == IGNORE          # padded region
    assert 0 not in short               # never the pad id
    assert batch["attention_mask"][1].tolist()[-1] == 0


def test_strip_prompt_returns_only_the_generation():
    assert strip_prompt("Twi units: abc\nTwi text: Ɔyɛ ne ho") == "Ɔyɛ ne ho"
    # A generation with no marker is returned as-is rather than emptied.
    assert strip_prompt("just text") == "just text"


def test_truncation_keeps_the_prompt_marker():
    """Truncating the finished prompt instead of the units drops the trailing
    "Twi text:" marker, leaving the model no cue for where its answer starts
    and `strip_prompt` nothing to split on."""
    tok = FakeTok()
    ex = build_example(tok, "a" * 500, "b" * 500, max_source_len=10, max_target_len=20)
    prompt_ids = [i for i, l in zip(ex["input_ids"], ex["labels"]) if l == IGNORE]
    assert tok.decode(prompt_ids).endswith("Twi text:")
    kept = [l for l in ex["labels"] if l != IGNORE]
    assert len(kept) == 21          # 20 target + EOS


def test_causal_scoring_left_pads_and_strips_the_prompt():
    """A decoder-only model continues its prompt.

    Right padding would insert pad tokens between the prompt and the first
    generated token, and the prompt is echoed back in the output, so it has to
    be sliced off before scoring.
    """
    import inspect

    from ghana_pico_asr.recovery.evaluate import score_causal

    src = inspect.getsource(score_causal)
    assert 'tok.padding_side = "left"' in src
    assert 'enc["input_ids"].shape[1]' in src      # slices off the echoed prompt
    assert "strip_prompt" in src
    # Padding side is restored even if generation raises.
    assert "finally:" in src


def test_causal_and_seq2seq_scorers_report_the_same_keys():
    """The comparison table only works if both paths report the same metrics."""
    import inspect

    from ghana_pico_asr.recovery.evaluate import score_causal, score_model

    keys = lambda f: set(  # noqa: E731
        m.group(1) for m in __import__("re").finditer(r'"(\w+)":', inspect.getsource(f))
    )
    for k in ("n", "cer", "wer", "exact_match", "baseline_cer", "samples"):
        assert k in keys(score_model) and k in keys(score_causal), k


def test_causal_models_get_no_extra_task_prefix():
    """`causal.PROMPT` already carries the instruction.

    Adding the T5 prefix on top desyncs training from inference: training
    would see "Twi units: restore twi: ..." while `score_causal` rebuilds the
    prompt from the unprefixed units.
    """
    import io as _io

    src = _io.open("job/hf_recovery.py", encoding="utf-8").read()
    assert '"" if (is_nllb or is_causal) else "restore twi: "' in src


def test_generation_tokenisation_matches_training():
    """`build_example` uses add_special_tokens=False, so generation must too.

    Gemma prepends <bos> by default; Qwen and SmolLM2 prepend nothing. Relying
    on the default meant gemma trained without a BOS and generated with one,
    producing runaway repetition (CER 9.12) while its neighbours were fine —
    invisible until a third tokeniser family appeared.
    """
    import inspect

    from ghana_pico_asr.recovery import causal
    from ghana_pico_asr.recovery.evaluate import score_causal

    assert "add_special_tokens=False" in inspect.getsource(causal.build_example)
    assert "add_special_tokens=False" in inspect.getsource(score_causal)
