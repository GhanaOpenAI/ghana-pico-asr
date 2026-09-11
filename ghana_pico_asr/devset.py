"""Held-out dev scoring during training, against human transcripts.

Validation UER is measured against the *aligner's* labels, so it inherits that
reference's noise floor and correlates only moderately with what we actually
care about: on the 30-epoch run, Spearman between validation balanced accuracy
and validation UER was -0.51, and the two metrics disagreed about the best
epoch by 0.012 UER.

This scores a slice of the Waxal dev shard instead -- whole utterances, human
transcripts, a corpus the model never trains on -- and produces `dev_uer`,
which can be used as the checkpoint-selection metric. Shard 1 stays untouched
for final reporting, so selecting on shard 0 does not contaminate it.

Deliberately small (a couple of hundred utterances): it runs every epoch, and
at ~5 s of audio each that is seconds of GPU time against a ~23 min epoch.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import features as F
from .infer import UnitTagger, surviving_runs
from .languages import flatten, get_language


def load_dev_set(
    lang_code: str,
    vocab: list[str],
    n_utts: int,
    cache_dir: str | None = None,
    min_units: int = 5,
) -> list[tuple[np.ndarray, list[int]]]:
    """Return ``(log-mel, reference unit ids)`` for the dev slice.

    Decoded once and held in memory as mel rather than audio: the features are
    what the model consumes, and recomputing them every epoch would waste more
    time than the scoring itself.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    lang = get_language(lang_code)
    idx = {u: i for i, u in enumerate(vocab)}
    logmel = F.LogMel(device="cpu")

    local = hf_hub_download(
        C.EVAL_REPO, C.EVAL_FILE, repo_type="dataset", cache_dir=cache_dir
    )
    out: list[tuple[np.ndarray, list[int]]] = []
    for batch in pq.ParquetFile(local).iter_batches(
        batch_size=8, columns=["audio", C.EVAL_TEXT_COL]
    ):
        if len(out) >= n_utts:
            break
        for row in batch.to_pylist():
            if len(out) >= n_utts:
                break
            text = (row.get(C.EVAL_TEXT_COL) or "").strip()
            if not text:
                continue
            ref = [idx[u] for u in flatten(lang.segment(text)) if u in idx]
            if len(ref) < min_units:
                continue
            wav = F.decode_audio(row["audio"])
            mel = logmel(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)))
            out.append((mel.numpy().astype(np.float32), ref))
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance, iterative with a single row."""
    if not a:
        return len(b)
    prev = list(range(len(a) + 1))
    for j, bj in enumerate(b, 1):
        cur = [j]
        for i, ai in enumerate(a, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (ai != bj)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def dev_uer(
    model,
    dev: list[tuple[np.ndarray, list[int]]],
    mean: float,
    std: float,
    device: str,
    sil_class: int,
    smooth_frames: int = 7,
    min_frames: int = 4,
) -> dict:
    """Corpus-level unit error rate over the dev slice.

    Uses the same smoothing and run-filtering as the released decoder, via
    `surviving_runs`, so the metric that selects a checkpoint matches the
    decoder that will run it.
    """
    was_training = model.training
    model.eval()
    total_err = total_ref = total_hyp = 0

    for mel, ref in dev:
        x = torch.from_numpy((mel - mean) / std).T.unsqueeze(0).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x.to(device))
        probs = torch.softmax(logits.float(), 1)[0].T.cpu().numpy()
        probs = UnitTagger.smooth(probs, smooth_frames)
        hyp = [
            cls
            for cls, _i, _n, _c in surviving_runs(
                probs, min_frames=min_frames, drop_class=sil_class
            )
        ]
        total_err += edit_distance(hyp, ref)
        total_ref += len(ref)
        total_hyp += len(hyp)

    if was_training:
        model.train()
    return {
        "dev_uer": total_err / max(total_ref, 1),
        "dev_len_ratio": total_hyp / max(total_ref, 1),
        "dev_utts": len(dev),
    }
