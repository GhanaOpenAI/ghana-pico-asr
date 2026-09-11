"""Stage 2 data layer: fixed-length chunks with a label for every frame.

The feature store holds whole-utterance log-mels plus a unit/frame table. This
module turns that table into a **dense per-frame label track** — one int16 per
10 ms frame, stored as a memmapped blob parallel to the mel blob — and then
serves fixed-length chunks of (mel, labels).

Doing it that way means the label track is computed once per configuration
rather than per epoch, and ``__getitem__`` is two array slices.

Frame labels:

* a unit's frames get that unit's class;
* frames covered by no unit (pauses, breaths, leading/trailing silence) get
  ``<sil>`` — class 0, which streaming inference needs somewhere to put them;
* frames of units the aligner was unsure about, or that fell below the
  vocabulary frequency floor, get ``IGNORE_INDEX`` and contribute no loss.
  They are *not* silently mislabelled as silence.

Splits are made **by utterance**, never by chunk, since overlapping chunks from
one utterance share audio.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from .prepare import UNIT_INVENTORY


# --------------------------------------------------------------------------- #
# Feature store access
# --------------------------------------------------------------------------- #


def list_shards(root: str, splits=("tts", "asr")) -> list[tuple[str, str]]:
    """Return ``(mel_path, npz_path)`` for every prepared shard, sorted."""
    man_dir = os.path.join(root, C.MANIFEST_DIR)
    mel_dir = os.path.join(root, C.MEL_DIR)
    if not os.path.isdir(man_dir):
        return []
    out = []
    for name in sorted(os.listdir(man_dir)):
        if not name.endswith(".npz"):
            continue
        stem = name[: -len(".npz")]
        if stem.split("_")[0] not in splits:
            continue
        mel = os.path.join(mel_dir, stem + ".npy")
        if os.path.exists(mel):
            out.append((mel, os.path.join(man_dir, name)))
    return out


def shard_stem(npz_path: str) -> str:
    return os.path.basename(npz_path)[: -len(".npz")]


def store_fingerprint(shards: list[tuple[str, str]]) -> str:
    """Identify the feature store by its shard names and byte sizes.

    Cache keys must depend on the *data*, not just the config: after stage 1
    adds or rewrites shards, a vocabulary or label track cached from an earlier
    (smaller) store must not be silently reused.
    """
    h = hashlib.sha1()
    for mel_path, npz_path in shards:
        h.update(shard_stem(npz_path).encode())
        for path in (mel_path, npz_path):
            try:
                h.update(str(os.path.getsize(path)).encode())
            except OSError:
                h.update(b"missing")
    return h.hexdigest()[:12]


def cache_key(root: str, cfg: C.ChunkConfig) -> str:
    return f"{cfg.key()}_{store_fingerprint(list_shards(root, cfg.splits))}"


def _utt_hash_bucket(stem: str, utt_i: int) -> float:
    """Deterministic [0, 1) bucket for utterance-level train/val/test split."""
    h = hashlib.md5(f"{stem}:{utt_i}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


def build_vocab(root: str, cfg: C.ChunkConfig) -> dict:
    """Count unit *frames* across the corpus and keep what is frequent enough.

    Counting frames rather than occurrences is the right measure here: the loss
    is per frame, so a unit's weight in training is its total duration.
    """
    frames: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    for _, npz_path in list_shards(root, cfg.splits):
        with np.load(npz_path) as z:
            uids = z["unit_ids"]
            dur = (z["ends"] - z["starts"]).astype(np.int64)
            keep = z["scores"].astype(np.float32) >= cfg.min_unit_score
        for uid, d in zip(uids[keep].tolist(), dur[keep].tolist()):
            frames[UNIT_INVENTORY[uid]] += d
            occurrences[UNIT_INVENTORY[uid]] += 1

    units = sorted(u for u, n in occurrences.items() if n >= cfg.min_unit_count)
    vocab = [C.SIL_TOKEN] + units
    return {
        "units": vocab,
        "unit_to_class": {u: i for i, u in enumerate(vocab)},
        "frame_counts": {u: int(n) for u, n in frames.most_common()},
        "occurrence_counts": {u: int(n) for u, n in occurrences.most_common()},
        "config": cfg.to_dict(),
    }


def vocab_path(root: str, cfg: C.ChunkConfig) -> str:
    return os.path.join(root, C.INDEX_DIR, f"vocab_{cache_key(root, cfg)}.json")


def load_or_build_vocab(root: str, cfg: C.ChunkConfig) -> dict:
    path = vocab_path(root, cfg)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    vocab = build_vocab(root, cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)
    # Also write the stable path, as the human-readable current vocabulary.
    # Best-effort: the authoritative copy is the keyed one above, and a feature
    # store can legitimately be read-only -- a published dataset mounted into a
    # training job is exactly that -- where a convenience write must not take
    # the run down with it.
    stable = os.path.join(root, C.VOCAB_PATH)
    try:
        os.makedirs(os.path.dirname(stable), exist_ok=True)
        with open(stable, "w", encoding="utf-8") as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[data] could not refresh {stable} ({exc.strerror}); "
              "the keyed vocabulary is authoritative", flush=True)
    return vocab


# --------------------------------------------------------------------------- #
# Dense frame-label tracks
# --------------------------------------------------------------------------- #


def label_dir(root: str, cfg: C.ChunkConfig) -> str:
    return os.path.join(root, C.INDEX_DIR, f"labels_{cache_key(root, cfg)}")


def build_label_track(root: str, cfg: C.ChunkConfig, vocab: dict) -> dict:
    """Write one int16 label-per-frame blob per shard; return the chunk index."""
    inv_to_class = np.full(len(UNIT_INVENTORY), C.IGNORE_INDEX, dtype=np.int16)
    for i, unit in enumerate(UNIT_INVENTORY):
        if unit in vocab["unit_to_class"]:
            inv_to_class[i] = vocab["unit_to_class"][unit]

    gap_label = np.int16(C.IGNORE_INDEX if cfg.silence_as_ignore else 0)
    min_gap = max(1, cfg.min_silence_gap_ms // C.FRAME_MS)
    chunk = cfg.chunk_frames
    stride = cfg.stride_frames

    shards = list_shards(root, cfg.splits)
    if not shards:
        raise RuntimeError(f"no prepared shards under {root}")
    out_dir = label_dir(root, cfg)
    os.makedirs(out_dir, exist_ok=True)

    idx_shard: list[np.ndarray] = []
    idx_start: list[np.ndarray] = []
    idx_valid: list[np.ndarray] = []
    idx_bucket: list[np.ndarray] = []
    stats: Counter[str] = Counter()

    for shard_i, (mel_path, npz_path) in enumerate(shards):
        stem = shard_stem(npz_path)
        mel_blob = np.load(mel_path, mmap_mode="r")
        with np.load(npz_path) as z:
            unit_ids = z["unit_ids"]
            u_start = z["starts"].astype(np.int64)
            u_end = z["ends"].astype(np.int64)
            u_score = z["scores"].astype(np.float32)
            ptr = z["utt_unit_ptr"]
            offsets = z["utt_offsets"].astype(np.int64)
            nframes = z["utt_nframes"].astype(np.int64)
            utt_scores = z["utt_scores"].astype(np.float32)
            total_frames = int(z["total_frames"][0])

        track = np.full(total_frames, C.IGNORE_INDEX, dtype=np.int16)

        for utt_i in range(len(offsets)):
            off = int(offsets[utt_i])
            n_f = int(nframes[utt_i])
            if utt_scores[utt_i] < cfg.min_utt_score:
                stats["utt_low_score"] += 1
                continue  # leave the whole utterance as IGNORE

            lo, hi = int(ptr[utt_i]), int(ptr[utt_i + 1])
            cls = inv_to_class[unit_ids[lo:hi]]
            cls = np.where(
                u_score[lo:hi] >= cfg.min_unit_score, cls, np.int16(C.IGNORE_INDEX)
            ).astype(np.int16)

            # Everything starts unlabelled; units paint over it. A frame left
            # unpainted is a gap, and `covered` remembers which those are so a
            # unit dropped for a low score stays IGNORE rather than becoming
            # silence.
            track[off : off + n_f] = C.IGNORE_INDEX
            covered = np.zeros(n_f, dtype=bool)
            for k in range(hi - lo):
                a, b = int(u_start[lo + k]), int(u_end[lo + k])
                track[off + a : off + b] = cls[k]
                covered[a:b] = True

            # A gap becomes <sil> only if it is both long enough to be a real
            # pause and quiet enough not to be untranscribed speech.
            if gap_label != C.IGNORE_INDEX:
                gaps = _runs_of_false(covered)
                if gaps:
                    energy = np.asarray(
                        mel_blob[off : off + n_f], dtype=np.float32
                    ).mean(axis=1)
                    speech_energy = energy[covered]
                    if len(speech_energy) >= 20 and cfg.silence_energy_percentile < 100:
                        quiet_below = float(
                            np.percentile(speech_energy, cfg.silence_energy_percentile)
                        )
                    elif cfg.silence_energy_percentile >= 100:
                        quiet_below = np.inf  # check disabled
                    else:
                        # No speech reference in this utterance, so we cannot
                        # tell pause from speech. Claim nothing.
                        quiet_below = -np.inf

                    for a, b in gaps:
                        if b - a < min_gap:
                            stats["short_gap_frames_ignored"] += b - a
                            continue
                        quiet = np.flatnonzero(energy[a:b] < quiet_below) + a
                        track[off + quiet] = gap_label
                        stats["sil_frames"] += len(quiet)
                        stats["loud_gap_frames_ignored"] += (b - a) - len(quiet)

            # Chunk starts covering the utterance; the last one is pulled back
            # so the tail is seen, and short utterances yield one padded chunk.
            if n_f >= chunk:
                starts = list(range(0, n_f - chunk + 1, stride))
                if starts[-1] != n_f - chunk:
                    starts.append(n_f - chunk)
                valid = [chunk] * len(starts)
            else:
                starts = [0]
                valid = [n_f]

            # A train-only split is pinned above every split threshold, so it
            # can never land in validation or test. Used where one corpus
            # re-transcribes another's audio: without this, the same audio
            # could be trained on under one transcript and evaluated under the
            # other, inflating validation.
            bucket = (
                1.0
                if stem.split("_")[0] in C.TRAIN_ONLY_SPLITS
                else _utt_hash_bucket(stem, utt_i)
            )
            idx_shard.append(np.full(len(starts), shard_i, dtype=np.int32))
            idx_start.append(off + np.asarray(starts, dtype=np.int64))
            idx_valid.append(np.asarray(valid, dtype=np.int32))
            idx_bucket.append(np.full(len(starts), bucket, dtype=np.float32))
            stats["utts"] += 1
            stats["chunks"] += len(starts)
            stats["labelled_frames"] += int((track[off : off + n_f] != C.IGNORE_INDEX).sum())
            stats["frames"] += n_f

        np.save(os.path.join(out_dir, stem + ".npy"), track)

    if not idx_shard:
        raise RuntimeError(
            "every utterance was filtered out while building the label track "
            f"({dict(stats)}). Loosen min_utt_score / min_unit_score."
        )

    index = {
        "shard": np.concatenate(idx_shard),
        "frame_start": np.concatenate(idx_start),
        "valid_len": np.concatenate(idx_valid),
        "bucket": np.concatenate(idx_bucket),
        "shards": shards,
        "label_dir": out_dir,
        "stats": dict(stats),
        "chunk_frames": chunk,
        "stride_frames": stride,
    }

    if cfg.max_chunks_per_split:
        index = _cap_per_split(index, cfg.max_chunks_per_split)
    return index


def _runs_of_false(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans where ``mask`` is False."""
    if not len(mask):
        return []
    padded = np.concatenate(([True], mask, [True]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _cap_per_split(index: dict, cap: int) -> dict:
    """Randomly subsample chunks so no corpus dominates."""
    rng = np.random.default_rng(0)
    stems = np.array([shard_stem(p).split("_")[0] for _, p in index["shards"]])
    split_of = stems[index["shard"]]
    keep = []
    for split in np.unique(split_of):
        sel = np.flatnonzero(split_of == split)
        if len(sel) > cap:
            sel = rng.choice(sel, size=cap, replace=False)
        keep.append(sel)
    keep = np.sort(np.concatenate(keep))
    out = dict(index)
    for key in ("shard", "frame_start", "valid_len", "bucket"):
        out[key] = index[key][keep]
    out["stats"] = {**index["stats"], "chunks_after_cap": int(len(keep))}
    return out


def index_cache_path(root: str, cfg: C.ChunkConfig) -> str:
    return os.path.join(root, C.INDEX_DIR, f"chunks_{cache_key(root, cfg)}.npz")


def load_or_build_index(root: str, cfg: C.ChunkConfig, vocab: dict) -> dict:
    path = index_cache_path(root, cfg)
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            return {
                "shard": z["shard"],
                "frame_start": z["frame_start"],
                "valid_len": z["valid_len"],
                "bucket": z["bucket"],
                "shards": [tuple(p) for p in z["shards"].tolist()],
                "label_dir": str(z["label_dir"]),
                "stats": json.loads(str(z["stats"])),
                "chunk_frames": int(z["chunk_frames"]),
                "stride_frames": int(z["stride_frames"]),
            }

    index = build_label_track(root, cfg, vocab)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        shard=index["shard"],
        frame_start=index["frame_start"],
        valid_len=index["valid_len"],
        bucket=index["bucket"],
        shards=np.array(index["shards"], dtype=object),
        label_dir=index["label_dir"],
        stats=json.dumps(index["stats"]),
        chunk_frames=index["chunk_frames"],
        stride_frames=index["stride_frames"],
    )
    return index


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class ChunkDataset(Dataset):
    """Serves ``(mel[1, N_MELS, T], labels[T])`` with ``T = chunk_frames``.

    Blobs are memmapped lazily per worker — a memmap cannot be forked safely,
    so each DataLoader worker opens its own handles on first access.
    """

    def __init__(
        self,
        index: dict,
        subset: np.ndarray | None = None,
        mean: float = 0.0,
        std: float = 1.0,
        augment: bool = False,
        train_cfg: C.TrainConfig | None = None,
    ):
        self.shards = index["shards"]
        self.label_dir = index["label_dir"]
        self.chunk_frames = index["chunk_frames"]

        sel = slice(None) if subset is None else subset
        self.shard_id = index["shard"][sel]
        self.frame_start = index["frame_start"][sel]
        self.valid_len = index["valid_len"][sel]

        self.mean = float(mean)
        self.std = float(std) if std else 1.0
        self.augment = augment
        self.tc = train_cfg or C.TrainConfig()
        self._mels: list | None = None
        self._labels: list | None = None

    def __len__(self) -> int:
        return len(self.frame_start)

    def _open(self, shard_i: int):
        if self._mels is None:
            self._mels = [None] * len(self.shards)
            self._labels = [None] * len(self.shards)
        if self._mels[shard_i] is None:
            mel_path, npz_path = self.shards[shard_i]
            self._mels[shard_i] = np.load(mel_path, mmap_mode="r")
            self._labels[shard_i] = np.load(
                os.path.join(self.label_dir, shard_stem(npz_path) + ".npy"), mmap_mode="r"
            )
        return self._mels[shard_i], self._labels[shard_i]

    def _spec_augment(self, mel: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Frequency and time masking. Time-masked frames are also excluded
        from the loss — teaching the model to name a unit whose audio has just
        been blanked out would be teaching it to guess."""
        tc = self.tc
        n_mels, n_t = mel.shape
        for _ in range(tc.n_masks):
            if tc.freq_mask > 0:
                f = int(torch.randint(0, tc.freq_mask + 1, (1,)).item())
                if f:
                    f0 = int(torch.randint(0, max(1, n_mels - f), (1,)).item())
                    mel[f0 : f0 + f, :] = 0.0
            if tc.time_mask > 0:
                t = int(torch.randint(0, tc.time_mask + 1, (1,)).item())
                if t:
                    t0 = int(torch.randint(0, max(1, n_t - t), (1,)).item())
                    mel[:, t0 : t0 + t] = 0.0
                    labels[t0 : t0 + t] = C.IGNORE_INDEX
        return mel

    def __getitem__(self, i: int):
        s = int(self.frame_start[i])
        n = int(self.valid_len[i])
        T = self.chunk_frames
        mel_blob, lab_blob = self._open(int(self.shard_id[i]))

        mel = np.zeros((T, C.N_MELS), dtype=np.float32)
        lab = np.full(T, C.IGNORE_INDEX, dtype=np.int64)
        mel[:n] = mel_blob[s : s + n]
        lab[:n] = lab_blob[s : s + n]

        x = torch.from_numpy(mel.T.copy())  # [n_mels, T]
        x = (x - self.mean) / self.std
        y = torch.from_numpy(lab)
        if self.augment:
            x = self._spec_augment(x, y)
        return x.unsqueeze(0), y


def split_subsets(index: dict, tc: C.TrainConfig) -> dict[str, np.ndarray]:
    """Utterance-level train/val/test masks, from the per-chunk bucket value."""
    bucket = index["bucket"]
    val_hi = tc.val_frac
    test_hi = tc.val_frac + tc.test_frac
    return {
        "val": np.flatnonzero(bucket < val_hi),
        "test": np.flatnonzero((bucket >= val_hi) & (bucket < test_hi)),
        "train": np.flatnonzero(bucket >= test_hi),
    }


def estimate_norm(index: dict, n_chunks: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Global log-mel mean/std over a random sample of chunks."""
    rng = np.random.default_rng(seed)
    n = len(index["frame_start"])
    pick = rng.choice(n, size=min(n_chunks, n), replace=False)
    total = total_sq = 0.0
    count = 0
    shard_ids = index["shard"][pick]
    starts = index["frame_start"][pick]
    lens = index["valid_len"][pick]
    for shard_i in np.unique(shard_ids):
        blob = np.load(index["shards"][int(shard_i)][0], mmap_mode="r")
        m = shard_ids == shard_i
        for s, n_f in zip(starts[m], lens[m]):
            a = np.asarray(blob[int(s) : int(s) + int(n_f)], dtype=np.float64)
            total += a.sum()
            total_sq += (a**2).sum()
            count += a.size
    mean = total / count
    var = max(total_sq / count - mean * mean, 1e-8)
    return float(mean), float(np.sqrt(var))


def load_or_estimate_norm(root: str, index: dict, cfg: C.ChunkConfig) -> tuple[float, float]:
    path = os.path.join(root, C.INDEX_DIR, f"norm_{cache_key(root, cfg)}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d["mean"], d["std"]
    mean, std = estimate_norm(index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"mean": mean, "std": std}, fh, indent=2)
    return mean, std


def unit_median_frames(root: str, cfg: C.ChunkConfig, vocab: dict) -> list[float]:
    """Median span length, in mel frames, for each class.

    Used at decode time to split a long run into repeated units: collapsing
    identical frame labels cannot otherwise recover a geminate ("nnipa") or a
    long vowel ("hyɛɛ"), since two identical adjacent units look exactly like
    one long one.
    """
    units = vocab["units"]
    per_class: dict[int, list[np.ndarray]] = {}
    for _, npz_path in list_shards(root, cfg.splits):
        with np.load(npz_path) as z:
            uids = z["unit_ids"]
            dur = (z["ends"] - z["starts"]).astype(np.int32)
            keep = z["scores"].astype(np.float32) >= cfg.min_unit_score
        uids, dur = uids[keep], dur[keep]
        for uid in np.unique(uids):
            unit = UNIT_INVENTORY[uid]
            ci = vocab["unit_to_class"].get(unit)
            if ci is not None:
                per_class.setdefault(ci, []).append(dur[uids == uid])

    out = [0.0] * len(units)
    for ci, chunks in per_class.items():
        out[ci] = float(np.median(np.concatenate(chunks)))
    return out


def frame_class_counts(index: dict, subset: np.ndarray, n_classes: int) -> np.ndarray:
    """Label histogram over the frames actually used, for class weighting."""
    counts = np.zeros(n_classes, dtype=np.int64)
    shard_ids = index["shard"][subset]
    starts = index["frame_start"][subset]
    lens = index["valid_len"][subset]
    for shard_i in np.unique(shard_ids):
        _, npz_path = index["shards"][int(shard_i)]
        track = np.load(
            os.path.join(index["label_dir"], shard_stem(npz_path) + ".npy"), mmap_mode="r"
        )
        m = shard_ids == shard_i
        for s, n_f in zip(starts[m], lens[m]):
            seg = np.asarray(track[int(s) : int(s) + int(n_f)])
            seg = seg[seg >= 0]
            if len(seg):
                counts += np.bincount(seg, minlength=n_classes)
    return counts
