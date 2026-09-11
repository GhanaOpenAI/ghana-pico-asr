"""Build real (units, text) pairs locally. Maintainer tool.

For Twi this is not the route: `modal_app/make_pairs.py` transcribes the whole
805 h corpus from the stored mel spectrograms in ~30 min and publishes the
result to the Hub, which is where `pico finetune-data` reads it from.

This exists for a language whose pipeline does not exist yet — point it at any
Hugging Face dataset with audio and text, and it produces the same CSV shape on
one local GPU, so a text-recovery model can be bootstrapped before there is a
feature store to transcribe.

The pairs' unit side carries the recogniser's actual error profile, which is
what a text-recovery model must learn to repair rather than trust.

    python scripts/build_real_pairs.py <org>/<dataset> \
        -c model/best.pt -n 2000 -o real_pairs_<lang>.csv
"""

from __future__ import annotations

import sys
import time

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghana_pico_asr.cli._common import (  # noqa: E402
    add_decode_args,
    decode_kwargs,
    resolve_checkpoint,
)


def add_args(ap) -> None:
    ap.add_argument("dataset", help="HF dataset id with audio and a transcript column")
    ap.add_argument("-c", "--checkpoint", default=None)
    ap.add_argument("--config", default=None, help="dataset config/subset")
    ap.add_argument("--split", default="train")
    ap.add_argument("-n", "--limit", type=int, default=1000, help="0 means all")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--audio-column", default=None)
    ap.add_argument("--text-column", default=None)
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument(
        "-o", "--output", default=None, help=".csv or .jsonl out (default: stdout)"
    )
    ap.add_argument(
        "--min-units",
        type=int,
        default=5,
        help="skip utterances whose reference is shorter than this (default: 5)",
    )
    ap.add_argument(
        "--max-uer",
        type=float,
        default=1.0,
        help="skip pairs worse than this UER — a pair the recogniser got almost "
        "entirely wrong teaches noise, not repair (default: 1.0, keep all)",
    )
    ap.add_argument("--quiet", action="store_true")
    add_decode_args(ap)


def run(args) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("this needs:  pip install datasets") from None

    from ghana_pico_asr import config as C
    from ghana_pico_asr.cli.hf_dataset import (
        _find_audio_column,
        _find_text_column,
        _to_waveform,
    )
    from ghana_pico_asr.infer import UnitTagger
    from ghana_pico_asr.languages import flatten
    from ghana_pico_asr.pairs import Pair, write_csv, write_pairs
    from ghana_pico_asr.trainer import edit_distance

    ckpt = resolve_checkpoint(args.checkpoint)
    tagger = UnitTagger(ckpt, device=args.device)
    lang = tagger.language

    streaming = not args.no_streaming
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=streaming)
    audio_col = _find_audio_column(ds.features, args.audio_column)
    text_col = _find_text_column(ds.features, args.text_column)
    if not text_col:
        raise SystemExit(
            "make-pairs needs a transcript column to pair against.\n"
            f"  columns: {list(ds.features)}\n"
            "  Pass --text-column NAME."
        )

    if not args.quiet:
        print(
            f"[pico] {args.dataset} audio={audio_col!r} text={text_col!r} "
            f"| model {lang.code}, {tagger.receptive_field_ms} ms",
            file=sys.stderr,
        )

    if args.offset:
        ds = ds.skip(args.offset) if streaming else ds.select(range(args.offset, len(ds)))
    if args.limit:
        ds = ds.take(args.limit) if streaming else ds.select(range(min(args.limit, len(ds))))

    kw = decode_kwargs(args)
    pairs: list[Pair] = []
    skipped = {"short": 0, "high_uer": 0, "failed": 0, "no_text": 0}
    tot_err = tot_ref = 0
    t0 = time.time()

    for i, row in enumerate(ds):
        text = (row.get(text_col) or "").strip()
        if not text:
            skipped["no_text"] += 1
            continue
        ref = flatten(lang.segment(text))
        if len(ref) < args.min_units:
            skipped["short"] += 1
            continue
        try:
            wav = _to_waveform(row[audio_col], C.SAMPLE_RATE)
            units = [u.unit for u in tagger.units(wav, **kw)]
        except Exception as exc:  # noqa: BLE001
            skipped["failed"] += 1
            if not args.quiet:
                print(f"[pico] failed {args.offset + i}: {exc}", file=sys.stderr)
            continue

        err = edit_distance(units, ref)
        uer = err / len(ref)
        if uer > args.max_uer:
            skipped["high_uer"] += 1
            continue
        tot_err += err
        tot_ref += len(ref)
        pairs.append(
            Pair(
                units=" ".join(units),
                text=text,
                origin="real",
                source=args.dataset,
                id=str(args.offset + i),
                reference_units=" ".join(ref),
            )
        )
        if not args.quiet and len(pairs) % 100 == 0:
            print(f"[pico] {len(pairs)} pairs", file=sys.stderr)

    if args.output and args.output.lower().endswith(".csv"):
        write_csv(pairs, args.output)
    else:
        write_pairs(pairs, args.output)
    if not args.quiet:
        print(
            f"[pico] {len(pairs):,} real pairs in {time.time() - t0:.1f}s"
            + (f" | corpus UER {tot_err / tot_ref:.4f}" if tot_ref else "")
            + f" | skipped {dict((k, v) for k, v in skipped.items() if v)}",
            file=sys.stderr,
        )
        if args.output:
            print(f"[pico] -> {args.output}", file=sys.stderr)
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
