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
