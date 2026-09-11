"""What a checkpoint records about how it was made.

A published model outlives the conversation that produced it. Without this,
someone fine-tuning a year from now cannot tell which corpora it saw, which
aligner produced its labels, or which filters were applied — all of which
change what the weights mean.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone

from . import config as C


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, cwd=os.path.dirname(__file__)
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return None


def corpus_summary(root: str, splits: tuple[str, ...]) -> dict:
    """Per-source utterance/unit/hour counts and mean alignment score."""
    import numpy as np

    from . import dataset as D

    out: dict[str, dict] = {}
    for split in splits:
        shards = D.list_shards(root, (split,))
        if not shards:
            continue
        utts = units = frames = 0
        wsum = 0.0
        for _, npz in shards:
            with np.load(npz) as z:
                n = len(z["utt_offsets"])
                utts += n
                units += len(z["unit_ids"])
                frames += int(z["total_frames"][0])
                wsum += float(z["utt_scores"].astype(np.float32).sum())
        out[split] = {
            "utterances": int(utts),
            "units": int(units),
            "hours": round(frames * C.FRAME_MS / 1000 / 3600, 2),
            "mean_align_score": round(wsum / max(utts, 1), 4),
            "shards": len(shards),
        }
    return out


SOURCES = {
    "tts": {
        "hf_dataset": C.TTS_REPO,
        "text_column": C.TTS_TEXT_COL,
        "transcript": "human",
        "domain": "read speech (studio)",
    },
    "asr": {
        "hf_dataset": C.ASR_REPO,
        "text_column": C.ASR_TEXT_COL,
        "transcript": "machine (Google Gemini)",
        "domain": "health talk shows",
    },
    "kuma": {
        "hf_dataset": C.KUMA_REPO,
        "text_column": C.KUMA_TEXT_COL,
        "transcript": "machine (Google STT, ak)",
        "domain": "film dialogue",
    },
}

# The corpora added after the first three live in `EXTRA_SOURCES`, so they are
# folded in from there rather than restated: a corpus registered in the config
# but missing here would otherwise be published as `"hf_dataset": "unknown"` in
# a released checkpoint's provenance.
SOURCES.update(
    {
        name: {
            "hf_dataset": spec["repo"],
            "text_column": spec["text_col"],
            "transcript": spec.get("transcript", "unknown"),
            "domain": spec.get("note", ""),
            "source_id": C.SOURCE_IDS.get(name),
        }
        for name, spec in C.EXTRA_SOURCES.items()
    }
)

for _name, _spec in SOURCES.items():
    _spec.setdefault("source_id", C.SOURCE_IDS.get(_name))


def build(root: str, ccfg, tcfg, lang_code: str, extra: dict | None = None) -> dict:
    """Assemble the provenance block stored inside a checkpoint."""
    splits = tuple(ccfg.splits)
    prov = {
        "project": C.PROJECT,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": lang_code,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "labels_from": {
            "aligner": C.ALIGNER_MODEL,
            "method": "CTC forced alignment of grapheme units",
            "emission_stride_ms": C.ALIGNER_STRIDE_MS,
        },
        "sources": {s: SOURCES.get(s, {"hf_dataset": "unknown"}) for s in splits},
        "corpus": corpus_summary(root, splits),
        "label_policy": ccfg.to_dict(),
        "features": {
            "sample_rate": C.SAMPLE_RATE,
            "n_fft": C.N_FFT,
            "hop_length": C.HOP_LENGTH,
            "n_mels": C.N_MELS,
            "f_min": C.F_MIN,
            "f_max": C.F_MAX,
            "frame_ms": C.FRAME_MS,
        },
        "training": tcfg.to_dict(),
    }
    if extra:
        prov.update(extra)
    return prov


def describe(ckpt: dict) -> str:
    """Human-readable one-screen summary of a checkpoint, for `--info`."""
    p = ckpt.get("provenance") or {}
    lines = [
        f"project        {p.get('project', '?')}",
        f"language       {ckpt.get('language', p.get('language', '?'))}",
        f"classes        {ckpt.get('n_classes', '?')}",
        f"context        {ckpt.get('receptive_field_ms', '?')} ms",
        f"created        {p.get('created_utc', '?')}",
        f"git            {(p.get('git_commit') or '?')[:12]}"
        + (" (dirty)" if p.get("git_dirty") else ""),
    ]
    tc = ckpt.get("train_config", {})
    if tc:
        lines.append(
            f"architecture   channels={tc.get('channels')} temporal_dim={tc.get('temporal_dim')} "
            f"dilations={tc.get('dilations')}"
        )
    if ckpt.get("val"):
        v = ckpt["val"]
        lines.append(
            "validation     "
            + "  ".join(
                f"{k}={v[k]:.4f}" for k in ("balanced_acc", "macro_f1", "unit_error_rate") if k in v
            )
        )
    corpus = p.get("corpus") or {}
    if corpus:
        total_h = sum(c.get("hours", 0) for c in corpus.values())
        lines.append(f"trained on     {total_h:.1f} h across {len(corpus)} source(s)")
        for name, c in corpus.items():
            src = (p.get("sources") or {}).get(name, {})
            lines.append(
                f"  {name:<6} {c.get('hours', 0):>7.1f} h  {c.get('utterances', 0):>7,} utts  "
                f"align {c.get('mean_align_score', 0):+.3f}  {src.get('hf_dataset', '?')}"
            )
    if p.get("labels_from"):
        lines.append(f"labels from    {p['labels_from'].get('aligner')}")
    return "\n".join(lines)


def to_json(ckpt: dict) -> str:
    return json.dumps(ckpt.get("provenance", {}), ensure_ascii=False, indent=2)
