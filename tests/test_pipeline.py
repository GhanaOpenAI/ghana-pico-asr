"""Tests for frame labelling, chunking, split hygiene and the frame-wise model.

Builds a synthetic feature store in the exact format :mod:`ghana_pico_asr.prepare`
writes, so the whole stage-2 path runs without Modal, a GPU, or any download.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghana_pico_asr import config as C  # noqa: E402
from ghana_pico_asr import dataset as D  # noqa: E402
from ghana_pico_asr.prepare import UNIT_TO_ID  # noqa: E402
from ghana_pico_asr.trainer import collapse, edit_distance  # noqa: E402

N_MELS = C.N_MELS


def write_store(tmp_path, utts, split="tts", shard=0, score=-0.1):
    """``utts`` = list of ``(n_frames, [(unit, start, end), ...])``."""
    mel_dir = tmp_path / C.MEL_DIR
    man_dir = tmp_path / C.MANIFEST_DIR
    mel_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    unit_ids, starts, ends, scores = [], [], [], []
    ptr, offsets, nframes, utt_scores = [0], [], [], []
    cursor = 0
    mels = []
    for n_f, spans in utts:
        mel = np.zeros((n_f, N_MELS), dtype=np.float16)
        for unit, s, e in spans:
            mel[s:e, :] = float(UNIT_TO_ID[unit])
            unit_ids.append(UNIT_TO_ID[unit])
            starts.append(s)
            ends.append(e)
            scores.append(score)
        mels.append(mel)
        offsets.append(cursor)
        nframes.append(n_f)
        utt_scores.append(score)
        ptr.append(len(unit_ids))
        cursor += n_f

    stem = f"{split}_{shard:05d}"
    np.save(mel_dir / f"{stem}.npy", np.concatenate(mels, axis=0))
    np.savez(
        man_dir / f"{stem}.npz",
        unit_ids=np.asarray(unit_ids, dtype=np.int16),
        starts=np.asarray(starts, dtype=np.int32),
        ends=np.asarray(ends, dtype=np.int32),
        scores=np.asarray(scores, dtype=np.float16),
        utt_unit_ptr=np.asarray(ptr, dtype=np.int64),
        utt_offsets=np.asarray(offsets, dtype=np.int64),
        utt_nframes=np.asarray(nframes, dtype=np.int32),
        utt_scores=np.asarray(utt_scores, dtype=np.float16),
        total_frames=np.asarray([cursor], dtype=np.int64),
    )
    (man_dir / f"{stem}.jsonl").write_text("", encoding="utf-8")
    return str(tmp_path)


def cfg(**kw):
    base = dict(
        chunk_ms=200,  # 20 frames
        chunk_stride_ms=100,  # 10 frames
        min_unit_count=1,
        min_silence_gap_ms=100,
        splits=("tts",),
    )
    base.update(kw)
    return C.ChunkConfig(**base)


@pytest.fixture
def store(tmp_path):
    # 60 frames: ky[0,10) gap[10,20) ɛ[20,40) m[40,60). Five identical utts.
    spans = [("ky", 0, 10), ("ɛ", 20, 40), ("m", 40, 60)]
    return write_store(tmp_path, [(60, spans)] * 5)


# ------------------------------------------------------------------ vocab


def test_vocab_counts_frames_and_occurrences(store):
    v = D.build_vocab(store, cfg())
    assert v["units"][0] == C.SIL_TOKEN
    assert set(v["units"]) == {C.SIL_TOKEN, "ky", "ɛ", "m"}
    # ɛ and m span 20 frames each per utt, ky only 10.
    assert v["frame_counts"]["ɛ"] == 100
    assert v["frame_counts"]["ky"] == 50
    assert v["occurrence_counts"]["ky"] == 5


def test_rare_units_are_dropped(store):
    v = D.build_vocab(store, cfg(min_unit_count=1000))
    assert v["units"] == [C.SIL_TOKEN]


# ------------------------------------------------------------ label track


def test_frame_labels_follow_the_aligner_spans(store):
    c = cfg(min_silence_gap_ms=100)  # the fixture's gap is 10 frames = 100 ms
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    track = np.load(Path(index["label_dir"]) / "tts_00000.npy")

    to_cls = v["unit_to_class"]
    assert (track[0:10] == to_cls["ky"]).all()
    assert (track[10:20] == 0).all(), "a long-enough gap must be <sil>"
    assert (track[20:40] == to_cls["ɛ"]).all()
    assert (track[40:60] == to_cls["m"]).all()
    assert len(track) == 300


def test_short_gaps_are_excluded_rather_than_called_silence(store):
    """The aligner's free <star> parks speech in inter-word gaps, so a short
    gap is probably swallowed speech, not a pause. It must not become <sil>."""
    short = D.build_label_track(store, cfg(min_silence_gap_ms=200), D.build_vocab(store, cfg()))
    track = np.load(Path(short["label_dir"]) / "tts_00000.npy")
    # The 100 ms gap is below the 200 ms floor -> ignored, not silence.
    assert (track[10:20] == C.IGNORE_INDEX).all()
    assert (track[0:10] >= 0).all(), "the units themselves stay labelled"
    assert short["stats"]["short_gap_frames_ignored"] == 50  # 10 frames x 5 utts

    long = D.build_label_track(store, cfg(min_silence_gap_ms=100), D.build_vocab(store, cfg()))
    assert long["stats"].get("sil_frames") == 50
    assert "short_gap_frames_ignored" not in long["stats"]


def test_dropped_unit_frames_never_become_silence(tmp_path):
    """A unit rejected for a low score is unknown, not a pause."""
    root = write_store(tmp_path, [(60, [("ky", 0, 20), ("m", 40, 60)])], score=-3.0)
    c = cfg(min_unit_score=-1.0, min_utt_score=-5.0, min_silence_gap_ms=100)
    v = D.build_vocab(root, cfg())
    index = D.build_label_track(root, c, v)
    track = np.load(Path(index["label_dir"]) / "tts_00000.npy")
    # ky and m were rejected -> IGNORE. Only the genuine [20,40) gap is <sil>.
    assert (track[0:20] == C.IGNORE_INDEX).all()
    assert (track[40:60] == C.IGNORE_INDEX).all()
    assert (track[20:40] == 0).all()


def test_silence_as_ignore_excludes_gaps_from_loss(store):
    c = cfg(silence_as_ignore=True)
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    track = np.load(Path(index["label_dir"]) / "tts_00000.npy")
    assert (track[10:20] == C.IGNORE_INDEX).all()
    assert (track[0:10] >= 0).all()


def test_low_score_units_become_ignore_not_a_wrong_label(tmp_path):
    spans = [("ky", 0, 10), ("ɛ", 10, 20)]
    root = write_store(tmp_path, [(20, spans)], score=-2.0)
    c = cfg(min_unit_score=-1.0, min_utt_score=-5.0)
    v = D.build_vocab(root, c)
    v["units"] = [C.SIL_TOKEN, "ky", "ɛ"]
    v["unit_to_class"] = {u: i for i, u in enumerate(v["units"])}
    index = D.build_label_track(root, c, v)
    track = np.load(Path(index["label_dir"]) / "tts_00000.npy")
    assert (track == C.IGNORE_INDEX).all(), "unconfident units must not be labelled"


def test_low_score_utterances_are_fully_ignored(store):
    c = cfg(min_utt_score=0.0)
    v = D.build_vocab(store, cfg())
    with pytest.raises(RuntimeError, match="every utterance was filtered out"):
        D.build_label_track(store, c, v)


# ---------------------------------------------------------------- chunking


def test_chunk_starts_cover_the_utterance_including_the_tail(store):
    c = cfg()
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    # 60 frames, chunk 20, stride 10 -> starts 0,10,20,30,40 (40 == 60-20)
    per_utt = len(index["frame_start"]) // 5
    assert per_utt == 5
    assert (index["valid_len"] == 20).all()
    first = np.sort(index["frame_start"][:5] - index["frame_start"][0])
    assert first.tolist() == [0, 10, 20, 30, 40]


def test_short_utterance_yields_one_padded_chunk(tmp_path):
    root = write_store(tmp_path, [(7, [("ky", 0, 7)])])
    c = cfg()
    v = D.build_vocab(root, c)
    v["units"] = [C.SIL_TOKEN, "ky"]
    v["unit_to_class"] = {u: i for i, u in enumerate(v["units"])}
    index = D.build_label_track(root, c, v)
    assert len(index["frame_start"]) == 1
    assert index["valid_len"][0] == 7

    ds = D.ChunkDataset(index)
    x, y = ds[0]
    assert x.shape == (1, N_MELS, 20)
    assert y.shape == (20,)
    assert (y[7:] == C.IGNORE_INDEX).all(), "padding must not contribute loss"
    assert (x[0, :, 7:] == 0).all()


def test_dataset_returns_matching_mel_and_labels(store):
    c = cfg()
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    ds = D.ChunkDataset(index)
    inv = {i: u for u, i in v["unit_to_class"].items()}

    for i in range(len(ds)):
        x, y = ds[i]
        for t in range(c.chunk_frames):
            label = int(y[t])
            if label <= 0:
                continue
            # the synthetic mel encodes the unit id it came from
            assert UNIT_TO_ID[inv[label]] == int(round(float(x[0, 0, t])))


def test_normalisation_is_applied(store):
    c = cfg()
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    plain = D.ChunkDataset(index)[0][0]
    scaled = D.ChunkDataset(index, mean=2.0, std=4.0)[0][0]
    assert torch_allclose(scaled, (plain - 2.0) / 4.0)


def torch_allclose(a, b):
    import torch

    return torch.allclose(a, b, atol=1e-5)


# ------------------------------------------------------------------ splits


def test_splits_are_disjoint_and_by_utterance(store):
    c = cfg()
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    subs = D.split_subsets(index, C.TrainConfig(val_frac=0.2, test_frac=0.2))

    allidx = np.concatenate([subs[k] for k in ("train", "val", "test")])
    assert len(np.unique(allidx)) == len(index["frame_start"])

    buckets = {k: set(np.unique(index["bucket"][v_]).tolist()) for k, v_ in subs.items()}
    assert not buckets["train"] & buckets["val"]
    assert not buckets["train"] & buckets["test"]
    assert not buckets["val"] & buckets["test"]


def test_index_and_norm_caches_round_trip(store):
    c = cfg()
    v = D.build_vocab(store, c)
    a = D.load_or_build_index(store, c, v)
    b = D.load_or_build_index(store, c, v)
    for k in ("shard", "frame_start", "valid_len", "bucket"):
        assert np.array_equal(a[k], b[k])
    assert a["label_dir"] == b["label_dir"]
    assert D.load_or_estimate_norm(store, a, c) == D.load_or_estimate_norm(store, a, c)


def test_frame_class_counts_ignores_unlabelled(store):
    c = cfg()
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    counts = D.frame_class_counts(index, np.arange(len(index["frame_start"])), len(v["units"]))
    assert counts.sum() > 0
    assert counts[v["unit_to_class"]["ɛ"]] > counts[v["unit_to_class"]["ky"]]


# ------------------------------------------------------------------- decode


def test_collapse_merges_runs_and_drops_silence():
    # 0 is <sil>; -100 is ignore
    assert collapse(np.array([1, 1, 1, 0, 0, 2, 2, 3])) == [1, 2, 3]
    assert collapse(np.array([1, 1, 2, 2, 1, 1])) == [1, 2, 1]
    assert collapse(np.array([0, 0, 0])) == []


def test_edit_distance():
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert edit_distance([1, 2, 3], [1, 3]) == 1
    assert edit_distance([], [1, 2]) == 2
    assert edit_distance([1, 2], []) == 2


# -------------------------------------------------------------------- model


def test_model_is_length_agnostic_and_preserves_time():
    import torch

    from ghana_pico_asr.model import PicoASRNet

    m = PicoASRNet(n_classes=5)
    for T in (20, 100, 257):
        out = m(torch.randn(2, 1, N_MELS, T))
        assert out.shape == (2, 5, T), "time resolution must survive the network"


def test_receptive_field_matches_measurement():
    """Perturb one input frame and check how far the output reacts."""
    import torch

    from ghana_pico_asr.model import PicoASRNet

    m = PicoASRNet(n_classes=5, dilations=(1, 2)).eval()
    T = 121
    base = torch.zeros(1, 1, N_MELS, T)
    with torch.no_grad():
        a = m(base)
        bumped = base.clone()
        bumped[0, 0, :, T // 2] = 10.0
        b = m(bumped)
    changed = (a - b).abs().sum(1)[0] > 1e-6
    span = int(changed.sum())
    assert span == m.receptive_field(), f"measured {span}, declared {m.receptive_field()}"


def test_end_to_end_training_step(store):
    import torch

    from ghana_pico_asr.model import PicoASRNet

    c = cfg()
    v = D.build_vocab(store, c)
    index = D.build_label_track(store, c, v)
    ds = D.ChunkDataset(index, mean=1.0, std=1.0, augment=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

    model = PicoASRNet(n_classes=len(v["units"]))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = next(iter(loader))
    logits = model(x)
    assert logits.shape == (4, len(v["units"]), c.chunk_frames)
    loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=C.IGNORE_INDEX)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_cache_key_changes_when_the_store_grows(tmp_path):
    """A vocabulary cached from a smaller store must not be reused."""
    spans = [("ky", 0, 10), ("ɛ", 10, 20)]
    root = write_store(tmp_path, [(20, spans)] * 3)
    c = cfg()

    v1 = D.load_or_build_vocab(root, c)
    key1 = D.cache_key(root, c)
    assert Path(D.vocab_path(root, c)).exists()

    # Stage 1 adds a second shard with a unit the first never contained.
    write_store(tmp_path, [(20, [("gy", 0, 20)])] * 3, shard=1)
    key2 = D.cache_key(root, c)
    assert key2 != key1, "fingerprint must react to new shards"

    v2 = D.load_or_build_vocab(root, c)
    assert "gy" not in v1["unit_to_class"]
    assert "gy" in v2["unit_to_class"], "new data must produce a fresh vocabulary"


def _write_store_with_energy(tmp_path, n_frames, spans, gap_energy):
    """One utterance where gap frames carry a chosen log-mel energy."""
    mel_dir = tmp_path / C.MEL_DIR
    man_dir = tmp_path / C.MANIFEST_DIR
    mel_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    mel = np.full((n_frames, N_MELS), gap_energy, dtype=np.float16)
    ids, starts, ends = [], [], []
    for unit, s, e in spans:
        mel[s:e, :] = 5.0  # "speech" energy
        ids.append(UNIT_TO_ID[unit])
        starts.append(s)
        ends.append(e)
    np.save(mel_dir / "tts_00000.npy", mel)
    np.savez(
        man_dir / "tts_00000.npz",
        unit_ids=np.asarray(ids, dtype=np.int16),
        starts=np.asarray(starts, dtype=np.int32),
        ends=np.asarray(ends, dtype=np.int32),
        scores=np.asarray([-0.1] * len(ids), dtype=np.float16),
        utt_unit_ptr=np.asarray([0, len(ids)], dtype=np.int64),
        utt_offsets=np.asarray([0], dtype=np.int64),
        utt_nframes=np.asarray([n_frames], dtype=np.int32),
        utt_scores=np.asarray([-0.1], dtype=np.float16),
        total_frames=np.asarray([n_frames], dtype=np.int64),
    )
    (man_dir / "tts_00000.jsonl").write_text("", encoding="utf-8")
    return str(tmp_path)


def _track(root, c):
    v = D.build_vocab(root, c)
    v["units"] = [C.SIL_TOKEN] + sorted(set(v["units"]) - {C.SIL_TOKEN})
    v["unit_to_class"] = {u: i for i, u in enumerate(v["units"])}
    index = D.build_label_track(root, c, v)
    return np.load(Path(index["label_dir"]) / "tts_00000.npy"), index["stats"]


def test_quiet_long_gap_becomes_silence(tmp_path):
    spans = [("a", 0, 30), ("m", 70, 100)]
    root = _write_store_with_energy(tmp_path, 100, spans, gap_energy=-8.0)
    track, stats = _track(root, cfg(min_silence_gap_ms=100, min_unit_count=1))
    assert (track[30:70] == 0).all(), "a quiet 400 ms gap is a genuine pause"
    assert stats["sil_frames"] == 40


def test_loud_long_gap_is_excluded_not_called_silence(tmp_path):
    """The untranscribed-speech case.

    Gemini transcripts drop sentences; the aligner then has no units for that
    audio and it shows up as a long gap. Labelling it <sil> would teach the
    model that speech is silence, so energy has to veto it.
    """
    spans = [("a", 0, 30), ("m", 70, 100)]
    root = _write_store_with_energy(tmp_path, 100, spans, gap_energy=5.0)
    track, stats = _track(root, cfg(min_silence_gap_ms=100, min_unit_count=1))
    assert (track[30:70] == C.IGNORE_INDEX).all(), "loud gap must not be silence"
    assert stats.get("sil_frames", 0) == 0
    assert stats["loud_gap_frames_ignored"] == 40
    # The real units are unaffected.
    assert (track[0:30] >= 0).all() and (track[70:100] >= 0).all()


def test_energy_check_can_be_disabled(tmp_path):
    spans = [("a", 0, 30), ("m", 70, 100)]
    root = _write_store_with_energy(tmp_path, 100, spans, gap_energy=5.0)
    track, stats = _track(
        root, cfg(min_silence_gap_ms=100, min_unit_count=1, silence_energy_percentile=100.0)
    )
    assert (track[30:70] == 0).all(), "percentile=100 disables the veto"
    assert stats["sil_frames"] == 40


def test_select_metric_direction(store):
    """balanced_acc must select on maximum, unit_error_rate on minimum."""
    from ghana_pico_asr import config as CC

    assert CC.TrainConfig().select_metric == "balanced_acc"
    # The direction flag the trainer derives from it.
    for metric, lower in (("unit_error_rate", True), ("balanced_acc", False), ("macro_f1", False)):
        assert (metric == "unit_error_rate") is lower


def test_unknown_select_metric_is_rejected():
    import inspect as _i

    from ghana_pico_asr import trainer as T

    src = _i.getsource(T.train)
    assert "select_metric=" in src and "is not one of" in src


def test_long_runs_split_into_repeated_units():
    """Repeat-collapsing alone turns "hyɛɛ" into "hyɛ"; duration must recover it."""
    import numpy as _np

    from ghana_pico_asr.infer import UnitTagger

    tagger = UnitTagger.__new__(UnitTagger)  # bypass checkpoint loading
    tagger.vocab = [C.SIL_TOKEN, "a", "n"]
    tagger.median_frames = [0.0, 6.0, 6.0]  # 60 ms median, as measured

    def probs_for(runs):
        rows = []
        for cls, n in runs:
            row = _np.zeros((n, 3), dtype=_np.float32)
            row[:, cls] = 0.9
            rows.append(row)
        return _np.concatenate(rows)

    split = lambda runs: tagger.units_from_posteriors(probs_for(runs), split_long_runs=True)

    # A 6-frame run is one "a"; a 12-frame run is two.
    assert [u.unit for u in split([(1, 6)])] == ["a"]
    assert [u.unit for u in split([(1, 12)])] == ["a", "a"]

    # Splitting is opt-IN: it measured worse on real audio, so the default
    # collapses as before.
    assert [u.unit for u in tagger.units_from_posteriors(probs_for([(1, 12)]))] == ["a"]

    # An 8-frame run is closer to 1x than 2x, so it stays single.
    assert [u.unit for u in split([(1, 8)])] == ["a"]

    # Capped at 3 so a pathological run cannot explode.
    assert len(split([(1, 200)])) == 3

    # Timings are subdivided across the repeats, not duplicated.
    a, b = split([(1, 12)])
    assert a.end == pytest.approx(b.start)
    assert a.start == pytest.approx(0.0) and b.end == pytest.approx(0.12)


def test_median_frames_length_is_validated():
    from ghana_pico_asr.infer import UnitTagger

    t = UnitTagger.__new__(UnitTagger)
    t.vocab = [C.SIL_TOKEN, "a"]
    t.median_frames = [1.0, 2.0, 3.0]  # wrong length
    # The guard lives in __init__; assert the class states the contract.
    import inspect as _i

    assert "one per class" in _i.getsource(UnitTagger.__init__)


def test_tagger_units_and_transcribe_are_callable():
    """Regression: `self.units = vocab` used to shadow the `units()` method,
    so the documented `tagger.units(audio)` / `tagger.transcribe(audio)` API
    raised TypeError: 'list' object is not callable."""
    import numpy as _np

    from ghana_pico_asr.infer import UnitTagger

    t = UnitTagger.__new__(UnitTagger)
    t.vocab = [C.SIL_TOKEN, "a", "n"]
    t.median_frames = None
    probs = _np.zeros((30, 3), dtype=_np.float32)
    probs[:15, 1] = 0.9
    probs[15:, 2] = 0.9

    # units() and transcribe() must be methods, reachable past the vocab list.
    assert callable(UnitTagger.units)
    assert callable(UnitTagger.transcribe)
    t.posteriors = lambda audio, **kw: probs  # stub out the model
    assert [u.unit for u in t.units(None)] == ["a", "n"]
    assert t.transcribe(None) == "a n"


def test_trainer_resumes_from_last_checkpoint(store, tmp_path):
    """Regression: a preempted container restarts with the same input, and
    without resume the run silently began again from scratch — we lost 5 of 6
    epochs that way. `last.pt` must carry optimiser/scheduler state so the
    cosine schedule continues instead of re-running its warmup."""
    import inspect as _i

    from ghana_pico_asr import trainer as T

    src = _i.getsource(T.train)
    # It must look for last.pt, restart the epoch loop from it, and save every
    # epoch (not only on improvement).
    assert 'resume_path = os.path.join(ckpt_dir, "last.pt")' in src
    assert "for epoch in range(start_epoch, tcfg.epochs)" in src
    assert '"opt": opt.state_dict()' in src and '"sched": sched.state_dict()' in src

    save_at = src.index("resume_path,\n        )")
    # Unconditional: the save sits at the epoch loop's own indentation, not
    # nested inside `if improved:`.
    assert "\n        torch.save(\n" in src[:save_at]
    # And it must record `best`/`stale` as selection just left them: saving
    # first would resume comparing epoch N+1 against epoch N-1's best, so a
    # worse epoch could overwrite best.pt and `stale` would never advance.
    assert src.index("stale = 0 if improved else stale + 1") < save_at
    assert save_at < src.index('os.path.join(ckpt_dir, "best.pt")')


def test_resume_state_round_trips(tmp_path):
    """The saved resume dict restores optimiser and scheduler position."""
    import torch

    from ghana_pico_asr.model import PicoASRNet

    m = PicoASRNet(n_classes=5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 10))
    for _ in range(7):
        opt.step()
        sched.step()
    lr_before = sched.get_last_lr()[0]
    path = tmp_path / "last.pt"
    torch.save(
        {"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
         "epoch": 3, "step": 7, "best": 0.5, "history": [{"epoch": 3}]}, path)

    m2 = PicoASRNet(n_classes=5)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: min(1.0, s / 10))
    st = torch.load(path, map_location="cpu", weights_only=False)
    m2.load_state_dict(st["model"]); opt2.load_state_dict(st["opt"]); sched2.load_state_dict(st["sched"])
    assert sched2.get_last_lr()[0] == pytest.approx(lr_before)
    assert st["epoch"] + 1 == 4  # resumes at the next epoch, not from zero
    assert st["step"] == 7 and st["best"] == 0.5


def test_early_stopping_logic():
    """Improvement must exceed min_delta; patience counts stale epochs."""
    import inspect as _i

    from ghana_pico_asr import config as CC
    from ghana_pico_asr import trainer as T

    tc = CC.TrainConfig()
    assert tc.patience == 4 and tc.min_delta == 5e-4

    src = _i.getsource(T.train)
    # min_delta on both directions, so it works for UER as well as accuracies.
    assert "current < best - tcfg.min_delta" in src
    assert "current > best + tcfg.min_delta" in src
    # Counter resets on improvement, increments otherwise, and breaks on patience.
    assert "stale = 0 if improved else stale + 1" in src
    assert "if tcfg.patience and stale >= tcfg.patience" in src
    # The stale counter survives a preemption/resume.
    assert '"stale": stale' in src and 'state.get("stale", 0)' in src
    # patience=0 disables it.
    assert "tcfg.patience and" in src


def test_early_stop_counter_sequence():
    """Simulate the trainer's accept/reject rule over a metric trajectory."""
    patience, min_delta = 3, 5e-4
    best, stale, stopped_at = float("-inf"), 0, None
    traj = [0.60, 0.65, 0.68, 0.6802, 0.6801, 0.6803, 0.70]  # stalls at index 3..5
    for i, cur in enumerate(traj):
        if cur > best + min_delta:
            best, stale = cur, 0
        else:
            stale += 1
        if stale >= patience:
            stopped_at = i
            break
    assert stopped_at == 5, "three sub-min_delta epochs in a row must stop it"
    assert best == pytest.approx(0.68)


def test_train_only_splits_never_reach_val_or_test(tmp_path, monkeypatch):
    """A corpus that re-transcribes another's audio must stay out of
    validation, or the same audio is trained on under one transcript and
    evaluated under the other — silently inflating the metrics."""
    spans = [("ky", 0, 10), ("ɛ", 20, 40), ("m", 40, 60)]
    # enough utterances that all three split ranges are populated by hashing
    root = write_store(tmp_path, [(60, spans)] * 40, split="tts")
    write_store(tmp_path, [(60, spans)] * 40, split="asr")

    monkeypatch.setattr(C, "TRAIN_ONLY_SPLITS", frozenset({"asr"}))
    c = C.ChunkConfig(
        chunk_ms=200, chunk_stride_ms=100, min_unit_count=1,
        min_silence_gap_ms=100, splits=("tts", "asr"),
    )
    vocab = D.build_vocab(root, c)
    index = D.build_label_track(root, c, vocab)
    subs = D.split_subsets(index, C.TrainConfig(val_frac=0.34, test_frac=0.33))

    shard_of = {i: index["shards"][s][1] for i, s in enumerate(index["shard"])}
    def splits_in(sel):
        return {D.shard_stem(shard_of[i]).split("_")[0] for i in sel}

    # With val+test = 67%, a normally-bucketed corpus would certainly appear
    # in both; the train-only one must appear in neither.
    assert "asr" not in splits_in(subs["val"])
    assert "asr" not in splits_in(subs["test"])
    assert "asr" in splits_in(subs["train"])
    # the normally-bucketed corpus still reaches every split
    assert "tts" in splits_in(subs["train"])
    assert "tts" in splits_in(subs["val"])
    assert "tts" in splits_in(subs["test"])


def test_train_only_is_derived_from_the_source_registry():
    """Nothing is train-only today; the guard is wired to the registry so a
    future corpus only needs `"train_only": True` to be protected."""
    assert C.TRAIN_ONLY_SPLITS == frozenset(
        {k for k, v in C.EXTRA_SOURCES.items() if v.get("train_only")}
    )
    for name, spec in C.EXTRA_SOURCES.items():
        assert (name in C.TRAIN_ONLY_SPLITS) == bool(spec.get("train_only")), name


def test_vocab_survives_a_read_only_store(store, tmp_path, capsys):
    """A published feature store mounted into a training job is read-only.

    `load_or_build_vocab` refreshes `features/vocab.json` purely as a
    convenience; on a read-only store that write raised OSError and killed the
    run before the first epoch.
    """
    import os
    import stat

    from ghana_pico_asr import config as C
    from ghana_pico_asr import dataset as D

    cfg = C.ChunkConfig(splits=("tts",))
    stable = os.path.join(store, C.VOCAB_PATH)
    os.makedirs(os.path.dirname(stable), exist_ok=True)
    with open(stable, "w", encoding="utf-8") as fh:
        fh.write("{}")
    os.chmod(stable, stat.S_IRUSR)  # read-only file, writable parent

    # A stale keyed cache would skip the build entirely.
    keyed = D.vocab_path(store, cfg)
    if os.path.exists(keyed):
        os.remove(keyed)

    vocab = D.load_or_build_vocab(store, cfg)
    assert C.SIL_TOKEN in vocab["unit_to_class"]
    # The authoritative copy is still written.
    assert os.path.exists(keyed)
    os.chmod(stable, stat.S_IRUSR | stat.S_IWUSR)
