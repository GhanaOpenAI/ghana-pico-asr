"""Grafting final-epoch weights onto a selected checkpoint's metadata."""

import pytest
import torch

from ghana_pico_asr.release import releasable_from_resume


def _best():
    return {
        "model": {"w": torch.zeros(3, 2), "b": torch.zeros(3)},
        "vocab": ["<sil>", "a", "b"],
        "n_classes": 3,
        "language": "twi",
        "norm": {"mean": 0.0, "std": 1.0},
        "feature_config": {"n_mels": 40},
        "epoch": 6,
        "select_metric": "balanced_acc",
        "val": {"balanced_acc": 0.777, "unit_error_rate": 0.322},
    }


def _last(epoch=7):
    return {
        "model": {"w": torch.ones(3, 2), "b": torch.ones(3)},
        "opt": {"state": "big"},
        "sched": {"last_epoch": epoch},
        "epoch": epoch,
        "step": 498160,
        "best": 0.777,
        "stale": 1,
        "history": [
            {"epoch": 6, "val_balanced_acc": 0.777, "val_unit_error_rate": 0.322},
            {"epoch": 7, "val_balanced_acc": 0.776, "val_unit_error_rate": 0.321},
        ],
    }


def test_takes_last_weights_and_best_metadata():
    out = releasable_from_resume(_best(), _last())
    assert torch.equal(out["model"]["w"], torch.ones(3, 2))
    # Everything inference needs survives.
    for k in ("vocab", "n_classes", "language", "norm", "feature_config"):
        assert k in out
    assert out["epoch"] == 7
    assert out["promoted_from"]["selected_epoch"] == 6


def test_drops_resume_only_state():
    out = releasable_from_resume(_best(), _last())
    for k in ("opt", "sched", "step", "stale", "best"):
        assert k not in out


def test_val_block_follows_the_promoted_epoch():
    out = releasable_from_resume(_best(), _last())
    assert out["val"]["unit_error_rate"] == 0.321
    assert out["val"]["balanced_acc"] == 0.776


def test_missing_history_row_keeps_selected_metrics():
    last = _last(epoch=9)  # no matching history row
    out = releasable_from_resume(_best(), last)
    assert out["epoch"] == 9
    assert out["val"]["unit_error_rate"] == 0.322


def test_refuses_a_different_architecture():
    last = _last()
    last["model"]["w"] = torch.ones(5, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        releasable_from_resume(_best(), last)

    last = _last()
    del last["model"]["b"]
    with pytest.raises(ValueError, match="architecture mismatch"):
        releasable_from_resume(_best(), last)


def test_requires_weights():
    with pytest.raises(ValueError, match="no 'model'"):
        releasable_from_resume(_best(), {"epoch": 7})


def test_every_epoch_is_archived_not_just_the_best():
    """`best.pt` is overwritten by the next improvement.

    On the 30-epoch run that destroyed the lowest-UER epoch before it could be
    scored on the held-out set, while the selection metric preferred an epoch
    0.012 UER worse. Selection is a guess; the archive is what makes it
    revisable.
    """
    import inspect as _i

    from ghana_pico_asr import config as C
    from ghana_pico_asr.trainer import train

    assert C.TrainConfig().keep_epoch_checkpoints is True

    src = _i.getsource(train)
    archive_at = src.index('f"epoch_{epoch:02d}.pt"')
    improved_at = src.index('if improved:\n            torch.save(payload')
    # Unconditional: the archive write must not sit inside the `if improved`
    # branch, or it saves exactly the epochs that already survive.
    assert archive_at < improved_at
    assert "if tcfg.keep_epoch_checkpoints:" in src
    # Both writes share one payload, so an archived epoch is directly loadable
    # by UnitTagger rather than being a bare state_dict.
    assert src.count("torch.save(payload") == 2
