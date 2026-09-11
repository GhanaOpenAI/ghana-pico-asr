"""Stage 2 on Modal: train the frame-wise 2D CNN on the prepared feature store.

    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/train.py
    modal run modal_app/train.py --epochs 2 --max-train-chunks 20000   # quick
    modal run modal_app/train.py --dilations 1,2,4,8                   # 430 ms context
    modal run modal_app/train.py --stats-only     # measured unit durations
    modal run modal_app/train.py --inspect-only   # frame/class distribution
"""

import json

import modal

from modal_app.common import VOLUMES, app, image, volume
from ghana_pico_asr import config as C

image = image.add_local_python_source("modal_app")


# memory: the chunk index and frame-class histogram scale with the corpus;
# a 2.85M-chunk run OOMed on the default request.
# retries: A10G containers get preempted, and the trainer resumes from its
# last epoch checkpoint rather than restarting from scratch.
@app.function(
    image=image,
    gpu="a10g",
    volumes=VOLUMES,
    timeout=60 * 60 * 10,
    memory=32768,
    cpu=8.0,
    retries=modal.Retries(max_retries=3, initial_delay=15.0),
)
def run_training(ccfg_kwargs: dict, tcfg_kwargs: dict) -> dict:
    from ghana_pico_asr.trainer import train

    summary = train(
        C.VOLUME_MOUNT,
        C.ChunkConfig(**ccfg_kwargs),
        C.TrainConfig(**tcfg_kwargs),
        device="cuda",
    )
    volume.commit()
    return summary


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60, cpu=4.0, memory=16384)
def inspect(ccfg_kwargs: dict) -> dict:
    """Frame-label distribution, without training."""
    import numpy as np

    from ghana_pico_asr import dataset as D

    ccfg = C.ChunkConfig(**ccfg_kwargs)
    vocab = D.load_or_build_vocab(C.VOLUME_MOUNT, ccfg)
    index = D.load_or_build_index(C.VOLUME_MOUNT, ccfg, vocab)
    units = vocab["units"]
    counts = D.frame_class_counts(index, np.arange(len(index["frame_start"])), len(units))
    volume.commit()

    total = max(int(counts.sum()), 1)
    st = index["stats"]
    return {
        "n_shards": len(index["shards"]),
        "n_utts": st.get("utts"),
        "n_chunks": int(len(index["frame_start"])),
        "audio_hours": round(st.get("frames", 0) * C.FRAME_MS / 1000 / 3600, 2),
        "frames_labelled_pct": round(
            100 * st.get("labelled_frames", 0) / max(st.get("frames", 1), 1), 2
        ),
        "n_classes": len(units),
        "labelled_frames_in_chunks": total,
        "index_stats": st,
        "class_distribution": {
            units[i]: {"frames": int(counts[i]), "pct": round(100 * counts[i] / total, 3)}
            for i in np.argsort(-counts)
            if counts[i]
        },
        "dropped_units": {
            u: n
            for u, n in vocab["occurrence_counts"].items()
            if u not in vocab["unit_to_class"]
        },
    }


@app.function(image=image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=16384)
def unit_stats(splits: tuple = ("tts", "asr")) -> dict:
    """Measured duration distribution of the aligner's unit spans.

    This is what the receptive field should span a few of, so that each frame's
    prediction sees the co-articulation around it.
    """
    import numpy as np

    from ghana_pico_asr import dataset as D
    from ghana_pico_asr.prepare import UNIT_INVENTORY

    out = {}
    for split in splits:
        shards = D.list_shards(C.VOLUME_MOUNT, (split,))
        if not shards:
            continue
        durs, uids, scores, utt_scores = [], [], [], []
        for _, npz_path in shards:
            with np.load(npz_path) as z:
                durs.append((z["ends"] - z["starts"]).astype(np.float32) * C.FRAME_MS)
                uids.append(z["unit_ids"])
                scores.append(z["scores"].astype(np.float32))
                utt_scores.append(z["utt_scores"].astype(np.float32))
        durs = np.concatenate(durs)
        uids = np.concatenate(uids)
        scores = np.concatenate(scores)
        utt_scores = np.concatenate(utt_scores)

        pct = np.percentile(durs, [5, 25, 50, 75, 90, 95, 99])
        median = float(np.median(durs)) or 1.0
        per_unit = {}
        for uid in np.unique(uids):
            d = durs[uids == uid]
            if len(d) < 20:
                continue
            sc = scores[uids == uid]
            per_unit[UNIT_INVENTORY[uid]] = {
                "n": int(len(d)),
                "mean_ms": round(float(d.mean()), 1),
                "median_ms": round(float(np.median(d)), 1),
                # How well the aligner could place this unit at all. English
                # letters land here: the Twi digraph segmenter mis-splits
                # English orthography, so those units align badly and are
                # mostly discarded by min_unit_score rather than mislabelled.
                "mean_score": round(float(sc.mean()), 3),
                "median_score": round(float(np.median(sc)), 3),
                "pct_dropped_at_-1.0": round(float(100 * (sc < -1.0).mean()), 1),
            }

        out[split] = {
            "n_units": int(len(durs)),
            "mean_ms": round(float(durs.mean()), 1),
            "percentiles_ms": {
                k: round(float(v), 1)
                for k, v in zip(["p5", "p25", "p50", "p75", "p90", "p95", "p99"], pct)
            },
            "mean_align_score": round(float(scores.mean()), 4),
            # Threshold choice should come from these, not from a guess: each
            # entry is the share of data a given cutoff would discard.
            "unit_score_percentiles": {
                f"p{q}": round(float(np.percentile(scores, q)), 3)
                for q in (1, 5, 10, 25, 50, 75, 90)
            },
            "utt_score_percentiles": {
                f"p{q}": round(float(np.percentile(utt_scores, q)), 3)
                for q in (1, 5, 10, 25, 50, 75, 90)
            },
            "pct_units_dropped_at": {
                f"{t}": round(float(100 * (scores < t).mean()), 2)
                for t in (-0.5, -1.0, -1.5, -2.0, -3.0)
            },
            "pct_utts_dropped_at": {
                f"{t}": round(float(100 * (utt_scores < t).mean()), 2)
                for t in (-0.5, -1.0, -1.2, -1.5, -2.0)
            },
            "units_per_receptive_field": {
                f"{ms}ms": round(ms / median, 2) for ms in (130, 270, 430, 750)
            },
            "per_unit": dict(sorted(per_unit.items(), key=lambda kv: -kv[1]["mean_ms"])),
        }
    return out


@app.local_entrypoint()
def main(
    # chunking / labelling
    chunk_ms: int = C.CHUNK_MS,
    chunk_stride_ms: int = C.CHUNK_STRIDE_MS,
    min_unit_score: float = -1.0,
    min_utt_score: float = -2.0,
    min_silence_gap_ms: int = 120,
    silence_energy_percentile: float = 10.0,
    min_unit_count: int = 200,
    silence_as_ignore: bool = False,
    max_chunks_per_split: int = 0,
    splits: str = "tts,asr",
    # model / training
    dilations: str = ",".join(str(d) for d in C.TEMPORAL_DILATIONS),
    temporal_dim: int = 192,
    channels: str = "32,64,128",
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 3e-4,
    class_weight_alpha: float = 0.3,
    max_train_chunks: int = 0,
    num_workers: int = 8,
    run_name: str = "twi2dcnn",
    select_metric: str = "balanced_acc",
    patience: int = 4,
    min_delta: float = 5e-4,
    inspect_only: bool = False,
    stats_only: bool = False,
):
    split_tuple = tuple(s.strip() for s in splits.split(",") if s.strip())
    ccfg_kwargs = {
        "chunk_ms": chunk_ms,
        "chunk_stride_ms": chunk_stride_ms,
        "min_unit_score": min_unit_score,
        "min_utt_score": min_utt_score,
        "min_silence_gap_ms": min_silence_gap_ms,
        "silence_energy_percentile": silence_energy_percentile,
        "min_unit_count": min_unit_count,
        "silence_as_ignore": silence_as_ignore,
        "max_chunks_per_split": max_chunks_per_split,
        "splits": split_tuple,
    }

    if stats_only:
        print(json.dumps(unit_stats.remote(split_tuple), ensure_ascii=False, indent=2))
        return
    if inspect_only:
        print(json.dumps(inspect.remote(ccfg_kwargs), ensure_ascii=False, indent=2))
        return

    tcfg_kwargs = {
        "dilations": tuple(int(d) for d in dilations.split(",") if d.strip()),
        "temporal_dim": temporal_dim,
        "channels": tuple(int(c) for c in channels.split(",") if c.strip()),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "class_weight_alpha": class_weight_alpha,
        "max_train_chunks": max_train_chunks,
        "num_workers": num_workers,
        "run_name": run_name,
        "select_metric": select_metric,
        "patience": patience,
        "min_delta": min_delta,
    }

    summary = run_training.remote(ccfg_kwargs, tcfg_kwargs)
    print("\n=== summary ===")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "test_per_class"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\ncheckpoint: modal volume get {C.VOLUME_NAME} {C.CKPT_DIR}/{run_name}/best.pt .")
