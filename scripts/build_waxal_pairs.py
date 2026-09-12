"""Build (units, text) pairs from the held-out Waxal set, for scoring stage 2.

The recovery models are otherwise evaluated on held-out pairs from the
*training* corpora, whose reference text is substantially machine transcript.
That measures agreement with Gemini and Google STT, not correctness.

Waxal shard 1 is the honest test and the one stage 1 reports: human
transcripts, a corpus neither stage trained on. Pairing the recogniser's real
output on that audio with those transcripts gives an **end-to-end** number --
audio through both stages, scored against what a person wrote.

Shard 0 stays free for tuning; shard 1 is only ever reported on.

    python scripts/build_waxal_pairs.py -o waxal_pairs.jsonl
    python scripts/build_waxal_pairs.py --push ghanaopenai/twi-waxal-recovery-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(checkpoint: str | None, n: int, shard: str, device: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ghana_pico_asr import config as C
    from ghana_pico_asr import features as F
    from ghana_pico_asr.cli._common import resolve_checkpoint
    from ghana_pico_asr.infer import UnitTagger
    from ghana_pico_asr.languages import flatten

    tagger = UnitTagger(resolve_checkpoint(checkpoint), device=device)
    lang = tagger.language
    local = hf_hub_download(C.EVAL_REPO, shard, repo_type="dataset")

    out: list[dict] = []
    for batch in pq.ParquetFile(local).iter_batches(
        batch_size=4, columns=["audio", C.EVAL_TEXT_COL]
    ):
        if n and len(out) >= n:
            break
        for row in batch.to_pylist():
            if n and len(out) >= n:
                break
            text = (row.get(C.EVAL_TEXT_COL) or "").strip()
            if not text:
                continue
            wav = F.decode_audio(row["audio"])
            units = [u.unit for u in tagger.units(wav)]
            if len(units) < 8:
                continue
            out.append(
                {
                    "units": " ".join(units),
                    "text": text,
                    # The aligner's view of the same text, so a clean pair can
                    # be built from this set too.
                    "reference_units": " ".join(flatten(lang.segment(text))),
                    "origin": "waxal",
                    "source": 0,
                    "id": f"waxal{len(out):05d}",
                }
            )
            if len(out) % 25 == 0:
                print(f"  {len(out)} utterances", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--checkpoint", default=None,
                    help="default: the released weights from the Hub")
    ap.add_argument("-n", "--limit", type=int, default=0, help="0 = the whole shard")
    ap.add_argument("--shard", default=None, help="default: the held-out shard 1")
    ap.add_argument("-o", "--out", default="waxal_pairs.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--push", default=None, metavar="REPO_ID")
    args = ap.parse_args(argv)

    from ghana_pico_asr import config as C

    shard = args.shard or C.EVAL_FILE_HELDOUT
    rows = build(args.checkpoint, args.limit, shard, args.device)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(rows)} pairs -> {args.out}")

    if rows:
        from ghana_pico_asr.recovery.data import pair_uer

        mean = sum(pair_uer(r["units"], r["reference_units"]) for r in rows) / len(rows)
        print(f"mean pair UER (recogniser vs human transcript): {mean:.4f}")

    if args.push:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, "_waxal.parquet", compression="zstd")
        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj="_waxal.parquet",
                        path_in_repo="data/train-00000-of-00001.parquet",
                        repo_id=args.push, repo_type="dataset",
                        commit_message="held-out Waxal pairs for stage-2 evaluation")
        os.remove("_waxal.parquet")
        print(f"pushed -> https://huggingface.co/datasets/{args.push}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
