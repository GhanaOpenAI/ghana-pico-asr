"""Stage 1: align one source parquet shard and write its feature store.

One call to :func:`process_shard` is one Modal task. It writes, per shard:

``features/mel/{split}_{shard:05d}.npy``
    All utterances' log-mels concatenated into a single ``[T_total, N_MELS]``
    float16 array, memory-mappable at train time.
``features/manifest/{split}_{shard:05d}.npz``
    Flat integer arrays describing where each utterance lives in that blob and
    which grapheme unit occupies which mel frames.
``features/manifest/{split}_{shard:05d}.jsonl``
    Human-readable per-utterance record (id, text, duration, scores) for
    debugging and spot-checking. Not read by the training pipeline.

Unit times are stored as **utterance-relative mel-frame indices** (10 ms), not
seconds, so the windowing code never has to re-round a float.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np
import torch

from . import config as C
from . import features as F
from .languages import get_language
from .align import AlignmentError, align_utterance, load_aligner

#: Stable unit -> id mapping, identical in every shard task. Derived from the
#: language inventory, so unit ids in the feature store stay meaningful.
def inventory(lang_code: str = C.LANGUAGE) -> list[str]:
    return get_language(lang_code).units


UNIT_INVENTORY: list[str] = inventory()
UNIT_TO_ID: dict[str, int] = {u: i for i, u in enumerate(UNIT_INVENTORY)}
_LANG = get_language()


def shard_filename(n_shards: int, shard: int) -> str:
    return f"data/train-{shard:05d}-of-{n_shards:05d}.parquet"


def select_asr_shards(
    n_shards: int = C.ASR_N_SHARDS, n_used: int = C.ASR_N_SHARDS_USED
) -> list[int]:
    """Evenly spaced shard indices, for speaker/topic spread across the repo."""
    return sorted(set(np.linspace(0, n_shards - 1, n_used).round().astype(int).tolist()))


def process_shard(
    repo_id: str,
    split: str,
    shard: int,
    n_shards: int,
    text_col: str,
    out_root: str,
    max_utts: int,
    device: str = "cuda",
    hf_cache: str | None = None,
    overwrite: bool = False,
    decode_backend: str = "ffmpeg",
) -> dict:
    """Align + featurise up to ``max_utts`` utterances from one parquet shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    mel_path = os.path.join(out_root, C.MEL_DIR, f"{split}_{shard:05d}.npy")
    npz_path = os.path.join(out_root, C.MANIFEST_DIR, f"{split}_{shard:05d}.npz")
    jsonl_path = os.path.join(out_root, C.MANIFEST_DIR, f"{split}_{shard:05d}.jsonl")
    for path in (mel_path, npz_path, jsonl_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(npz_path) and not overwrite:
        with np.load(npz_path) as z:
            return {
                "split": split,
                "shard": shard,
                "status": "cached",
                "n_utts": int(len(z["utt_offsets"])),
            }

    t0 = time.time()
    local = hf_hub_download(
        repo_id, shard_filename(n_shards, shard), repo_type="dataset", cache_dir=hf_cache
    )
    t_download = time.time() - t0

    model, tokenizer = load_aligner(device=device)
    logmel = F.LogMel(device=device)

    mels: list[np.ndarray] = []
    unit_ids: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    scores: list[float] = []
    utt_unit_ptr: list[int] = [0]
    utt_offsets: list[int] = []
    utt_nframes: list[int] = []
    utt_scores: list[float] = []
    records: list[dict] = []

    frame_cursor = 0
    skipped: Counter[str] = Counter()
    t0 = time.time()

    pf = pq.ParquetFile(local)
    done = False
    row_i = -1
    for batch in pf.iter_batches(batch_size=8, columns=["audio", text_col]):
        if done:
            break
        rows = batch.to_pylist()
        for row in rows:
            row_i += 1
            if len(utt_offsets) >= max_utts:
                done = True
                break

            text = (row.get(text_col) or "").strip()
            if not text:
                skipped["empty_text"] += 1
                continue

            try:
                # ffmpeg subprocess by default: an in-process libsndfile crash
                # on a malformed file is a SIGSEGV/SIGABRT that Python cannot
                # catch, and it killed entire shards. A subprocess failure is
                # just a non-zero exit code.
                wav = F.decode_audio(row["audio"], backend=decode_backend)
            except Exception as exc:  # noqa: BLE001 - a bad cell must not kill the shard
                skipped[f"decode:{type(exc).__name__}"] += 1
                continue

            duration = len(wav) / C.SAMPLE_RATE
            if not (C.MIN_DURATION_S <= duration <= C.MAX_DURATION_S):
                skipped["duration"] += 1
                continue

            words = _LANG.segment(text)
            n_units = sum(len(w) for w in words)
            if n_units < C.MIN_UNITS:
                skipped["too_few_units"] += 1
                continue
            ups = n_units / duration
            if not (C.MIN_UNITS_PER_SEC <= ups <= C.MAX_UNITS_PER_SEC):
                skipped["units_per_sec"] += 1
                continue

            try:
                spans = align_utterance(model, tokenizer, wav, text)
            except AlignmentError:
                skipped["align_failed"] += 1
                continue
            except Exception as exc:  # noqa: BLE001
                skipped[f"align:{type(exc).__name__}"] += 1
                continue

            mel = logmel(torch.from_numpy(wav)).cpu().numpy().astype(np.float16)
            n_frames = mel.shape[0]

            # Clamp unit frames into the mel blob; the aligner's 20 ms grid can
            # round one frame past the end of a 10 ms mel sequence.
            kept = 0
            for span in spans:
                s = int(round(span.start * 1000.0 / C.FRAME_MS))
                e = int(round(span.end * 1000.0 / C.FRAME_MS))
                s = max(0, min(s, n_frames - 1))
                e = max(s + 1, min(e, n_frames))
                unit_ids.append(UNIT_TO_ID[span.unit])
                starts.append(s)
                ends.append(e)
                scores.append(span.score)
                kept += 1

            mean_score = float(np.mean([sp.score for sp in spans]))
            mels.append(mel)
            utt_offsets.append(frame_cursor)
            utt_nframes.append(n_frames)
            utt_scores.append(mean_score)
            utt_unit_ptr.append(utt_unit_ptr[-1] + kept)
            frame_cursor += n_frames

            records.append(
                {
                    "utt": f"{split}_{shard:05d}_{row_i:06d}",
                    "row": row_i,
                    "duration": round(duration, 3),
                    "n_frames": n_frames,
                    "n_units": kept,
                    "mean_score": round(mean_score, 4),
                    "text": text,
                }
            )

    t_align = time.time() - t0

    if not mels:
        raise RuntimeError(f"{split} shard {shard}: no utterances survived filtering ({skipped})")

    blob = np.concatenate(mels, axis=0)
    del mels
    np.save(mel_path, blob)
    np.savez(
        npz_path,
        unit_ids=np.asarray(unit_ids, dtype=np.int16),
        starts=np.asarray(starts, dtype=np.int32),
        ends=np.asarray(ends, dtype=np.int32),
        scores=np.asarray(scores, dtype=np.float16),
        utt_unit_ptr=np.asarray(utt_unit_ptr, dtype=np.int64),
        utt_offsets=np.asarray(utt_offsets, dtype=np.int64),
        utt_nframes=np.asarray(utt_nframes, dtype=np.int32),
        utt_scores=np.asarray(utt_scores, dtype=np.float16),
        total_frames=np.asarray([blob.shape[0]], dtype=np.int64),
    )
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "split": split,
        "shard": shard,
        "status": "ok",
        "n_utts": len(utt_offsets),
        "n_units": len(unit_ids),
        "audio_hours": round(sum(r["duration"] for r in records) / 3600.0, 3),
        "mel_frames": int(blob.shape[0]),
        "mel_mb": round(blob.nbytes / 1e6, 1),
        "mean_score": round(float(np.mean(utt_scores)), 4),
        "skipped": dict(skipped),
        "sec_download": round(t_download, 1),
        "sec_align": round(t_align, 1),
    }
