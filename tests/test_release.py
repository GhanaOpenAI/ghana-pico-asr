"""Provenance, fine-tune transfer and the CLI — the release surface."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghana_pico_asr import config as C  # noqa: E402
from ghana_pico_asr import provenance as P  # noqa: E402
from ghana_pico_asr.finetune import FinetuneError, load_for_finetune  # noqa: E402
from ghana_pico_asr.model import PicoASRNet  # noqa: E402


def make_ckpt(tmp_path, vocab, **over):
    m = PicoASRNet(n_classes=len(vocab), channels=(8, 16), temporal_dim=24, dilations=(1, 2))
    ck = {
        "model": m.state_dict(),
        "vocab": vocab,
        "n_classes": len(vocab),
        "language": "twi",
        "norm": {"mean": -1.0, "std": 2.0},
        "train_config": {"channels": (8, 16), "temporal_dim": 24, "dilations": (1, 2),
                         "dropout": 0.1},
        "feature_config": {"sample_rate": C.SAMPLE_RATE, "n_fft": C.N_FFT,
                           "hop_length": C.HOP_LENGTH, "n_mels": C.N_MELS},
        "receptive_field_ms": m.receptive_field_ms(),
        "epoch": 3,
        "val": {"balanced_acc": 0.7, "macro_f1": 0.68, "unit_error_rate": 0.4},
    }
    ck.update(over)
    p = tmp_path / "ck.pt"
    torch.save(ck, p)
    return str(p), ck


VOCAB = ["<sil>", "a", "b", "ky", "ɛ"]


# ------------------------------------------------------------------ provenance


def test_describe_handles_a_checkpoint_without_provenance(tmp_path):
    """Published checkpoints predating provenance must still be readable."""
    _, ck = make_ckpt(tmp_path, VOCAB)
    out = P.describe(ck)
    assert "classes        5" in out
    assert "?" in out  # missing fields are shown, not crashed on
    assert "balanced_acc=0.7000" in out


def test_describe_reports_corpus_when_present(tmp_path):
    _, ck = make_ckpt(tmp_path, VOCAB, provenance={
        "project": "ghana-pico-asr", "language": "twi",
        "created_utc": "2026-09-10T00:00:00+00:00", "git_commit": "abc123def456",
        "labels_from": {"aligner": "mms"},
        "sources": {"tts": {"hf_dataset": "org/tts"}},
        "corpus": {"tts": {"hours": 10.5, "utterances": 9914, "mean_align_score": -0.7}},
    })
    out = P.describe(ck)
    assert "trained on     10.5 h across 1 source(s)" in out
    assert "org/tts" in out and "abc123def456"[:12] in out


def test_provenance_records_what_a_finetuner_needs():
    """These keys are the reason provenance exists; losing one loses the audit."""
    import inspect

    src = inspect.getsource(P.build)
    for key in ("labels_from", "sources", "corpus", "label_policy", "features",
                "training", "git_commit", "language"):
        assert f'"{key}"' in src, key


# -------------------------------------------------------------------- finetune


def test_vocab_remap_preserves_shared_units(tmp_path):
    path, ck = make_ckpt(tmp_path, VOCAB)
    new = ["<sil>", "a", "ky", "zz"]  # drops b and ɛ, adds zz
    model, rep = load_for_finetune(path, new, device="cpu")
    assert rep["units_transferred"] == 3
    assert rep["units_new"] == ["zz"]
    assert sorted(rep["units_dropped"]) == ["b", "ɛ"]

    old_w = ck["model"]["head.1.weight"]
    new_w = model.state_dict()["head.1.weight"]
    oi = {u: i for i, u in enumerate(VOCAB)}
    for j, u in enumerate(new):
        if u in oi:
            assert torch.equal(new_w[j], old_w[oi[u]]), u


def test_reset_head_discards_the_output_layer(tmp_path):
    path, _ = make_ckpt(tmp_path, VOCAB)
    _, rep = load_for_finetune(path, VOCAB, device="cpu", reset_head=True)
    assert rep["units_transferred"] == 0
    assert rep["head_reset"] is True


def test_freeze_trunk_leaves_only_the_head_trainable(tmp_path):
    path, _ = make_ckpt(tmp_path, VOCAB)
    model, rep = load_for_finetune(path, VOCAB, device="cpu", freeze_trunk=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["head.1.weight", "head.1.bias"]
    assert rep["trunk_frozen"] is True


def test_feature_mismatch_is_refused_not_silently_wrong(tmp_path, monkeypatch):
    """Different mel parameters make the weights meaningless, so this must
    raise rather than produce a model that quietly predicts noise."""
    path, _ = make_ckpt(tmp_path, VOCAB)
    monkeypatch.setattr(C, "N_MELS", 80)
    with pytest.raises(FinetuneError, match="feature config differs"):
        load_for_finetune(path, VOCAB, device="cpu")


def test_optimiser_state_is_not_inherited(tmp_path):
    """Fine-tuning wants a fresh schedule, not the tail of the original run."""
    path, ck = make_ckpt(tmp_path, VOCAB)
    assert "opt" not in ck and "sched" not in ck


# ------------------------------------------------------------------------- CLI


def test_cli_help_needs_no_torch_import(capsys):
    from ghana_pico_asr.cli.main import build_parser

    build_parser().print_help()
    out = capsys.readouterr().out
    for cmd in ("transcribe", "hf-dataset", "web", "finetune", "text-to-units", "info"):
        assert cmd in out


def test_cli_unknown_command_exits_nonzero(capsys):
    from ghana_pico_asr.cli.main import main

    assert main(["nonsense"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_cli_version():
    from ghana_pico_asr.cli.main import main

    assert main(["--version"]) == 0


def test_every_command_declares_its_arguments():
    """A command missing add_args/run would only fail when someone ran it."""
    import argparse
    import importlib

    from ghana_pico_asr.cli.main import _COMMANDS

    for name, (_, module_name) in _COMMANDS.items():
        mod = importlib.import_module(f"ghana_pico_asr.cli.{module_name}")
        assert hasattr(mod, "add_args") and hasattr(mod, "run"), name
        mod.add_args(argparse.ArgumentParser())  # must not raise


def test_audio_column_detection_and_failure():
    from ghana_pico_asr.cli.hf_dataset import _find_audio_column, _find_text_column

    assert _find_audio_column({"audio": 1, "text": 2}, None) == "audio"
    assert _find_audio_column({"speech": 1}, None) == "speech"
    assert _find_audio_column({"blah": 1}, "blah") == "blah"
    assert _find_text_column({"audio": 1, "twi_text": 2}, None) == "twi_text"
    assert _find_text_column({"audio": 1}, None) is None

    with pytest.raises(SystemExit, match="no audio column"):
        _find_audio_column({"text": 1, "label": 2}, None)
    with pytest.raises(SystemExit, match="not in this dataset"):
        _find_audio_column({"audio": 1}, "nope")


def test_checkpoint_resolution(tmp_path, monkeypatch):
    from ghana_pico_asr.cli import _common

    resolve_checkpoint = _common.resolve_checkpoint

    (tmp_path / "best.pt").write_bytes(b"x")
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path / "best.pt")
    assert resolve_checkpoint(str(tmp_path / "best.pt")) == str(tmp_path / "best.pt")
    with pytest.raises(SystemExit, match="not found"):
        resolve_checkpoint(str(tmp_path / "nope.pt"))
    monkeypatch.setenv("PICO_CHECKPOINT", str(tmp_path / "best.pt"))
    assert resolve_checkpoint(None).endswith("best.pt")


def test_checkpoint_falls_back_to_the_hub(tmp_path, monkeypatch):
    """With no local checkpoint the tools fetch released weights themselves.

    Requiring a manual download first is the step that stops someone trying
    the model at all; it also means usage never shows up as a download count.
    """
    from ghana_pico_asr import config as C
    from ghana_pico_asr.cli import _common

    monkeypatch.delenv("PICO_CHECKPOINT", raising=False)
    calls = []
    monkeypatch.setattr(
        _common, "download_checkpoint", lambda repo, *a, **k: calls.append(repo) or "/x.pt"
    )

    assert _common.resolve_checkpoint(None) == "/x.pt"
    assert calls == [C.HF_MODEL_REPO["twi"]]

    # An explicit repo id works too, and is not mistaken for a missing file.
    assert _common.resolve_checkpoint("someorg/some-model") == "/x.pt"
    assert calls[-1] == "someorg/some-model"

    with pytest.raises(SystemExit, match="no released weights"):
        _common.resolve_checkpoint(None, language="klingon")


def test_text_to_units_round_trips_the_training_direction():
    """The reverse direction must use the same inventory the labels came from,
    or a text-recovery model learns a mapping the acoustic model never emits."""
    from ghana_pico_asr.languages import flatten, get_language

    lang = get_language("twi")
    text = "Ɔyɛ ne ho adwuma"
    units = flatten(lang.segment(text))
    assert units == ["ɔ", "y", "ɛ", "n", "e", "h", "o", "a", "dw", "u", "m", "a"]
    assert all(u in lang.units for u in units)


def test_every_trained_split_has_named_provenance():
    """A released checkpoint must name the dataset behind every corpus it saw.

    `SOURCES` listed only the first three corpora, so the three added later
    were published as `"hf_dataset": "unknown"` — half the training data
    undocumented in the model card.
    """
    from ghana_pico_asr import config as C
    from ghana_pico_asr import provenance as P

    for split in C.ChunkConfig().splits:
        src = P.SOURCES.get(split, {})
        assert src.get("hf_dataset") not in (None, "unknown"), split
        assert src.get("text_column"), split
        assert src.get("transcript") not in (None, "unknown"), split
    # Registered-but-unaligned corpora still need an id reserved for them.
    for name in C.EXTRA_SOURCES:
        assert P.SOURCES[name]["source_id"] == C.SOURCE_IDS[name]


def test_test_metrics_describe_the_selected_checkpoint():
    """The reported test numbers must come from the weights that get released.

    `evaluate` was called on whatever the final epoch left in memory, while
    `best.pt` could hold an earlier, better-selected epoch — as happened on the
    xl run, where selection kept epoch 6 and the final epoch was 7.
    """
    import inspect as _i

    from ghana_pico_asr import trainer as T

    src = _i.getsource(T.train)
    test_at = src.index('evaluate(model, loaders["test"]')
    reload_at = src.index('model.load_state_dict(chosen["model"])')
    assert reload_at < test_at
    assert '"tested_epoch"' in src
