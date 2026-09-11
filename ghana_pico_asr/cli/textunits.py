"""Convert text to grapheme-unit sequences — the reverse of what the model does.

This is the same function that produced the model's training labels, so the two
directions share one inventory by construction. If they drifted, a text-recovery
model trained on this output would learn a mapping the acoustic model never
emits.

Its intended use is generating parallel data for a text-recovery model
(units -> sentence) without needing any audio: take raw text in a new domain,
convert it here, and you have the input side of the pair.

    pico text-to-units corpus.txt -o pairs.jsonl
    pico text-to-units --hf-dataset some/corpus --text-column text -n 50000
    echo "Ɔyɛ ne ho adwuma" | pico text-to-units -
"""

from __future__ import annotations

import sys

from ._common import write_records


def add_args(ap) -> None:
    ap.add_argument(
        "inputs",
        nargs="*",
        help="text files, one sentence per line; use - for stdin",
    )
    ap.add_argument("--hf-dataset", default=None, help="read text from an HF dataset instead")
    ap.add_argument("--config", default=None, help="dataset config/subset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", default=None, help="text column (auto-detected)")
    ap.add_argument("-n", "--limit", type=int, default=0, help="0 means all")
    ap.add_argument("-l", "--language", default=None, help="language code (default: twi)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument(
        "-f", "--format", default="jsonl", choices=("jsonl", "json", "csv", "text")
    )
    ap.add_argument(
        "--per-word",
        action="store_true",
        help="keep word boundaries as ' | ' instead of one flat sequence",
    )
    ap.add_argument(
        "--drop-unchanged",
        action="store_true",
        help="skip lines that normalise to nothing (all digits/punctuation)",
    )


def _iter_text(args):
    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit("--hf-dataset needs:  pip install datasets") from None
        from .hf_dataset import TEXT_HINTS

        ds = load_dataset(args.hf_dataset, args.config, split=args.split, streaming=True)
        col = args.text_column
        if not col:
            names = list(ds.features or {})
            col = next((h for h in TEXT_HINTS if h in names), None)
            if not col:
                raise SystemExit(f"no text column found; columns: {names}")
        if args.limit:
            ds = ds.take(args.limit)
        for row in ds:
            if row.get(col):
                yield str(row[col])
        return

    if not args.inputs:
        raise SystemExit("give text files, - for stdin, or --hf-dataset")
    n = 0
    for item in args.inputs:
        fh = sys.stdin if item == "-" else open(item, encoding="utf-8")
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield line
                n += 1
                if args.limit and n >= args.limit:
                    return
        finally:
            if item != "-":
                fh.close()


def run(args) -> int:
    from ..languages import flatten, get_language

    lang = get_language(args.language)
    records: list[dict] = []
    skipped = 0
    for text in _iter_text(args):
        words = lang.segment(text)
        if not words:
            skipped += 1
            if args.drop_unchanged:
                continue
        units = " | ".join(" ".join(w) for w in words) if args.per_word else " ".join(flatten(words))
        records.append(
            {
                "text": text,
                "units": units,
                "n_units": sum(len(w) for w in words),
                "n_words": len(words),
            }
        )

    write_records(records, args.output, args.format)
    print(
        f"[pico] {len(records)} lines converted ({lang.code})"
        + (f", {skipped} produced no units" if skipped else ""),
        file=sys.stderr,
    )
    return 0
