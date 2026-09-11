"""Build fine-tuning data for a new domain from raw text, with replay.

Takes text in a domain the recogniser has never heard, converts it to clean
grapheme units, and mixes in a sample of real audio-derived pairs so the
text-recovery model keeps its repair behaviour instead of learning that its
input is trustworthy.

The real pairs ship with the package, so this needs no audio, no GPU and no
preparatory step — one command turns raw text into training data.

    pico finetune-data corpus.txt -o mixed.jsonl
    pico finetune-data --hf-dataset some/legal-corpus --text-column body \
         -n 20000 --ratio 1.0 -o mixed.jsonl
    pico finetune-data corpus.txt --real ghananlpcommunity/twi-grapheme-unit-pairs \
         --balance equal -o mixed.jsonl
"""

from __future__ import annotations

import json
import sys


def add_args(ap) -> None:
    ap.add_argument("inputs", nargs="*", help="text files, one sentence per line; - for stdin")
    ap.add_argument("--hf-dataset", default=None, help="read new-domain text from an HF dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-column", default=None)
    ap.add_argument("-n", "--limit", type=int, default=0, help="cap new-domain lines; 0 = all")
    ap.add_argument(
        "--real",
        default=None,
        help="where to take real pairs from: a local .csv/.jsonl, or an HF "
        "dataset id like org/name. Default: the language's published set "
        "on the Hub",
    )
    ap.add_argument(
        "--real-limit",
        type=int,
        default=0,
        help="cap how many real pairs to load from HF (0 = all)",
    )
    ap.add_argument(
        "--balance",
        default="proportional",
        choices=("proportional", "equal"),
        help="how to spread the replay across source corpora. The corpus is "
        "unbalanced (film dialogue is ~75%%), so 'proportional' mirrors the "
        "real error profile while 'equal' stops one domain dominating "
        "(default: proportional)",
    )
    ap.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="parts real per part synthetic. 1.0 is an even mix; if fewer real "
        "pairs exist than requested, all of them are used (default: 1.0)",
    )
    ap.add_argument("-l", "--language", default=None)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument(
        "--min-units", type=int, default=5, help="skip short lines (default: 5)"
    )
    ap.add_argument(
        "--report", default=None, help="also write the mix report as JSON here"
    )


def run(args) -> int:
    from ..languages import flatten, get_language
    from ..pairs import Pair, describe_mix, load_real_pairs, mix, write_pairs
    from .textunits import _iter_text

    lang = get_language(args.language)

    if not args.inputs and not args.hf_dataset:
        raise SystemExit("give text files, - for stdin, or --hf-dataset")

    # New-domain synthetic pairs: clean units from the deterministic segmenter.
    source = args.hf_dataset or ",".join(args.inputs)
    synthetic: list[Pair] = []
    dropped = 0
    for text in _iter_text(args):
        units = flatten(lang.segment(text))
        if len(units) < args.min_units:
            dropped += 1
            continue
        synthetic.append(
            Pair(
                units=" ".join(units),
                text=text,
                origin="synthetic",
                source=source,
                id=str(len(synthetic)),
            )
        )
    if not synthetic:
        raise SystemExit("no usable new-domain lines produced any units")

    real, real_path = load_real_pairs(args.real, language=lang.code, limit=args.real_limit)
    real = [p for p in real if p.origin == "real"] or real
    if not real:
        raise SystemExit(f"{real_path} contains no pairs")

    mixed, report = mix(
        synthetic,
        real,
        ratio=args.ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        balance=args.balance,
    )
    report["language"] = lang.code
    report["new_domain_source"] = source
    report["real_pairs_file"] = real_path
    report["new_domain_dropped_short"] = dropped

    write_pairs(mixed, args.output)
    print(describe_mix(report), file=sys.stderr)
    if dropped:
        print(f"({dropped} new-domain lines were too short and were dropped)", file=sys.stderr)
    if args.output:
        print(f"-> {args.output}", file=sys.stderr)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"-> {args.report}", file=sys.stderr)
    return 0
