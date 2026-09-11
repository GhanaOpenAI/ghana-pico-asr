"""Transcribe a Hugging Face dataset in place, by name.

Points at any dataset, validates that it really has an audio column, and
transcribes a chosen number of samples. Useful for evaluating the model on a
corpus you have not aligned, or for producing unit sequences to train a
text-recovery model against the dataset's own transcripts.
"""

from __future__ import annotations

import sys
import time

from ._common import add_decode_args, decode_kwargs, resolve_checkpoint, write_records

#: Column names people actually use for audio, tried in order when the dataset
#: features do not declare an Audio type.
AUDIO_HINTS = ("audio", "wav", "speech", "input_audio", "audio_file", "file", "path")
#: Likely reference-transcript columns, recorded alongside for comparison.
TEXT_HINTS = ("text", "transcription", "transcript", "sentence", "twi_text", "normalized_text")


def add_args(ap) -> None:
    ap.add_argument("dataset", help="HF dataset id, e.g. ghananlpcommunity/ghana-speech-eval")
    ap.add_argument("-c", "--checkpoint", default=None, help="checkpoint file or directory")
    ap.add_argument("--config", default=None, help="dataset config/subset name")
    ap.add_argument("--split", default="train", help="dataset split (default: train)")
    ap.add_argument(
        "-n",
        "--limit",
        type=int,
        default=100,
        help="how many samples to transcribe; 0 means all (default: 100)",
    )
    ap.add_argument("--offset", type=int, default=0, help="skip this many samples first")
    ap.add_argument(
        "--audio-column",
        default=None,
        help="override audio column detection",
    )
    ap.add_argument(
        "--text-column",
        default=None,
        help="reference transcript column to record alongside (auto-detected)",
    )
    ap.add_argument(
        "--no-streaming",
        action="store_true",
        help="download the dataset instead of streaming it",
    )
    ap.add_argument("-o", "--output", default=None, help="write here instead of stdout")
    ap.add_argument(
        "-f", "--format", default="jsonl", choices=("jsonl", "json", "csv", "text")
    )
    ap.add_argument(
        "--score",
        action="store_true",
        help="if a text column is present, also report unit error rate against it",
    )
    ap.add_argument("--quiet", action="store_true")
    add_decode_args(ap)


def _find_audio_column(features, override: str | None) -> str:
    """Validate that an audio column exists, with an actionable error if not."""
    names = list(features)
    if override:
        if override not in names:
            raise SystemExit(
                f"--audio-column {override!r} is not in this dataset. Columns: {names}"
            )
        return override

    # Prefer a declared Audio feature — the only unambiguous signal.
    try:
        from datasets import Audio

        declared = [n for n, f in features.items() if isinstance(f, Audio)]
        if declared:
            return declared[0]
    except Exception:  # noqa: BLE001 - fall back to name hints
        pass

    for hint in AUDIO_HINTS:
        if hint in names:
            return hint

    raise SystemExit(
        "this dataset has no audio column.\n"
        f"  columns found: {names}\n"
        "  Pass --audio-column NAME if one of these holds audio, or pick a "
        "dataset with an Audio feature."
    )


def _find_text_column(features, override: str | None) -> str | None:
    names = list(features)
    if override:
        if override not in names:
            raise SystemExit(f"--text-column {override!r} not in dataset. Columns: {names}")
        return override
    for hint in TEXT_HINTS:
        if hint in names:
            return hint
    return None


def _to_waveform(cell, target_sr: int):
    """Coerce a dataset audio cell to mono float32 at ``target_sr``."""
    import io

    import numpy as np

    if isinstance(cell, dict):
        if cell.get("array") is not None:
            wav = np.asarray(cell["array"], dtype=np.float32)
            sr = int(cell.get("sampling_rate") or target_sr)
        elif cell.get("bytes"):
            import soundfile as sf

            wav, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        elif cell.get("path"):
            import soundfile as sf

            wav, sr = sf.read(cell["path"], dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        else:
            raise ValueError(f"unrecognised audio cell keys: {sorted(cell)}")
    elif isinstance(cell, str):
        import soundfile as sf

        wav, sr = sf.read(cell, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    else:
        raise ValueError(f"unrecognised audio cell type: {type(cell).__name__}")

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        import torch
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(np.ascontiguousarray(wav)), sr, target_sr).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def run(args) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "this mode needs the `datasets` package:  pip install datasets"
        ) from None

    from .. import config as C
    from ..infer import UnitTagger

    ckpt = resolve_checkpoint(args.checkpoint)
    tagger = UnitTagger(ckpt, device=args.device)

    streaming = not args.no_streaming
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=streaming)

    features = ds.features
    if not features:
        raise SystemExit(
            "could not read this dataset's features; try --no-streaming"
        )
    audio_col = _find_audio_column(features, args.audio_column)
    text_col = _find_text_column(features, args.text_column)

    if not args.quiet:
        print(
            f"[pico] {args.dataset}"
            + (f" ({args.config})" if args.config else "")
            + f" split={args.split}\n"
            f"[pico] audio column: {audio_col!r}"
            + (f" | reference text: {text_col!r}" if text_col else " | no reference text")
            + f"\n[pico] model: {tagger.language.code}, {tagger.receptive_field_ms} ms context",
            file=sys.stderr,
        )

    # Avoid decoding audio we are going to skip.
    if args.offset:
        ds = ds.skip(args.offset) if streaming else ds.select(range(args.offset, len(ds)))
    if args.limit:
        ds = ds.take(args.limit) if streaming else ds.select(range(min(args.limit, len(ds))))

    kw = decode_kwargs(args)
    records: list[dict] = []
    tot_err = tot_ref = 0
    t0 = time.time()

    for i, row in enumerate(ds):
        try:
            wav = _to_waveform(row[audio_col], C.SAMPLE_RATE)
            units = tagger.units(wav, **kw)
        except Exception as exc:  # noqa: BLE001
            print(f"[pico] FAILED sample {args.offset + i}: {exc}", file=sys.stderr)
            records.append({"index": args.offset + i, "units": "", "error": str(exc)})
            continue

        hyp = [u.unit for u in units]
        rec = {
            "index": args.offset + i,
            "units": " ".join(hyp),
            "n_units": len(hyp),
            "duration_s": round(len(wav) / C.SAMPLE_RATE, 3),
        }
        if units:
            rec["confidence"] = round(sum(u.confidence for u in units) / len(units), 4)
        if text_col and row.get(text_col):
            rec["reference_text"] = row[text_col]
            if args.score:
                from ..languages import flatten
                from ..trainer import edit_distance

                ref = flatten(tagger.language.segment(row[text_col]))
                rec["reference_units"] = " ".join(ref)
                e = edit_distance(hyp, ref)
                rec["uer"] = round(e / max(len(ref), 1), 4)
                tot_err += e
                tot_ref += len(ref)
        records.append(rec)
        if not args.quiet and (i + 1) % 25 == 0:
            print(f"[pico] {i + 1} samples", file=sys.stderr)

    write_records(records, args.output, args.format)
    if not args.quiet:
        msg = f"[pico] {len(records)} samples in {time.time() - t0:.1f}s"
        if args.score and tot_ref:
            msg += f" | corpus UER {tot_err / tot_ref:.4f} over {tot_ref:,} reference units"
        print(msg, file=sys.stderr)
    return 0
