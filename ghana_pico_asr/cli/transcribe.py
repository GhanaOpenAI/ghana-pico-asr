"""Transcribe audio to grapheme units — single file, batch, or HF dataset."""

from __future__ import annotations

import sys
import time

from ._common import (
    add_decode_args,
    decode_kwargs,
    iter_audio_paths,
    resolve_checkpoint,
    write_records,
)


def add_args(ap) -> None:
    ap.add_argument(
        "inputs",
        nargs="+",
        help="audio files and/or directories (directories are searched recursively)",
    )
    ap.add_argument("-c", "--checkpoint", default=None, help="checkpoint file or directory")
    ap.add_argument("-o", "--output", default=None, help="write here instead of stdout")
    ap.add_argument(
        "-f",
        "--format",
        default="text",
        choices=("text", "jsonl", "json", "csv"),
        help="output format (default: text)",
    )
    ap.add_argument(
        "--timings", action="store_true", help="include per-unit start/end/confidence"
    )
    ap.add_argument("--no-recursive", action="store_true", help="do not descend into directories")
    ap.add_argument("--quiet", action="store_true", help="suppress progress on stderr")
    add_decode_args(ap)


def run(args) -> int:
    from ..infer import UnitTagger

    ckpt = resolve_checkpoint(args.checkpoint)
    paths = iter_audio_paths(args.inputs, recursive=not args.no_recursive)
    if not paths:
        raise SystemExit("no audio files found")

    tagger = UnitTagger(ckpt, device=args.device)
    if not args.quiet:
        print(
            f"[pico] {tagger.language.code} | {len(tagger.vocab)} units | "
            f"{tagger.receptive_field_ms} ms context | {len(paths)} file(s)",
            file=sys.stderr,
        )

    kw = decode_kwargs(args)
    records: list[dict] = []
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        try:
            units = tagger.units(path, **kw)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
            print(f"[pico] FAILED {path}: {exc}", file=sys.stderr)
            records.append({"id": path, "units": "", "error": str(exc)})
            continue
        rec = {
            "id": path,
            "units": " ".join(u.unit for u in units),
            "n_units": len(units),
        }
        if units:
            rec["confidence"] = round(sum(u.confidence for u in units) / len(units), 4)
        if args.timings:
            rec["timings"] = [
                {
                    "unit": u.unit,
                    "start": round(u.start, 3),
                    "end": round(u.end, 3),
                    "confidence": round(u.confidence, 3),
                }
                for u in units
            ]
        records.append(rec)
        if not args.quiet and (i % 20 == 0 or i == len(paths)):
            print(f"[pico] {i}/{len(paths)}", file=sys.stderr)

    write_records(records, args.output, args.format)
    if not args.quiet:
        print(f"[pico] done in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0
