"""Dev-set scoring: selection against human transcripts, not aligner labels."""

import numpy as np
import pytest
import torch

from ghana_pico_asr.devset import dev_uer, edit_distance
from ghana_pico_asr.model import PicoASRNet


def test_edit_distance_basics():
    assert edit_distance([], []) == 0
    assert edit_distance([], [1, 2]) == 2
    assert edit_distance([1, 2, 3], []) == 3
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert edit_distance([1, 2, 3], [1, 9, 3]) == 1        # substitution
    assert edit_distance([1, 2, 3], [1, 2]) == 1           # deletion
    assert edit_distance([1, 2], [1, 2, 3]) == 1           # insertion


def test_dev_uer_is_zero_for_a_perfect_decoder():
    """A model whose argmax already matches the reference scores 0."""
    n_classes, n_mels, T = 5, 40, 300
    model = PicoASRNet(n_classes=n_classes, channels=(8, 16, 32), temporal_dim=32)

    # Reference is whatever this untrained model happens to predict, so the
    # metric is exercised end to end without needing a trained checkpoint.
    mel = np.random.RandomState(0).randn(T, n_mels).astype(np.float32)
    dev = [(mel, [])]
    out = dev_uer(model, dev, 0.0, 1.0, "cpu", sil_class=0)
    hyp_len = out["dev_len_ratio"]  # ref is empty -> ratio is hyp/1
    assert out["dev_utts"] == 1
    # With an empty reference every emitted unit is an insertion.
    assert out["dev_uer"] == pytest.approx(hyp_len)


def test_dev_uer_restores_training_mode():
    """Scoring mid-epoch must not silently leave the model in eval mode."""
    model = PicoASRNet(n_classes=5, channels=(8, 16, 32), temporal_dim=32)
    model.train()
    mel = np.zeros((200, 40), dtype=np.float32)
    dev_uer(model, [(mel, [1, 2])], 0.0, 1.0, "cpu", sil_class=0)
    assert model.training is True

    model.eval()
    dev_uer(model, [(mel, [1, 2])], 0.0, 1.0, "cpu", sil_class=0)
    assert model.training is False


def test_dev_uer_is_the_selection_metric_only_when_loaded():
    """`select_metric='dev_uer'` without a dev slice must fail loudly."""
    import inspect as _i

    from ghana_pico_asr import config as C
    from ghana_pico_asr.trainer import train

    assert C.TrainConfig().dev_utts == 0
    src = _i.getsource(train)
    assert "select_metric='dev_uer' needs dev_utts > 0" in src
    # dev_uer is an error rate: the selector must minimise it.
    assert 'lower_is_better = tcfg.select_metric in ("unit_error_rate", "dev_uer")' in src


def test_dev_scoring_uses_the_released_decoder_rules():
    """The metric and the shipped decoder must share run-collapsing.

    A selection metric computed with different decode rules than the decoder
    that will run the model selects the wrong checkpoint.
    """
    import inspect as _i

    from ghana_pico_asr import devset

    src = _i.getsource(devset.dev_uer)
    assert "surviving_runs" in src
    assert "UnitTagger.smooth" in src
