"""Shared CLI helpers: checkpoint discovery, audio loading, output writers."""

from __future__ import annotations

import json
import os
import sys
from typing import Iterable

DEFAULT_CKPT_NAMES = ("best.pt", "model.pt", "pico.pt")

#: Released weights per language, pulled from the Hub when nothing local is
#: given. Fetching through `huggingface_hub` is also what makes downloads
#: countable, so usage of the published model is visible.
from ghana_pico_asr import config as _C  # noqa: E402

HF_MODEL_REPO = _C.HF_MODEL_REPO
HF_WEIGHTS_FILE = _C.HF_WEIGHTS_FILE


def download_checkpoint(repo_id: str, filename: str = HF_WEIGHTS_FILE) -> str:
    """Fetch released weights from the Hub, caching them locally."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            f"fetching {repo_id} needs:  pip install huggingface_hub\n"
            "  (or pass --checkpoint PATH to use a local file)"
        ) from None
    try:
        return hf_hub_download(repo_id, filename)
    except Exception as exc:  # noqa: BLE001 - network, auth, missing repo
        raise SystemExit(
            f"could not fetch {filename} from {repo_id}: {exc}\n"
            "  Pass --checkpoint PATH to use a local file instead."
        ) from None


def resolve_checkpoint(path: str | None, language: str = "twi") -> str:
    """Accept a file, a directory, an HF repo id, $PICO_CHECKPOINT, or nothing.

    With nothing given, the language's released weights are pulled from the
    Hub, so the tool works on a fresh machine with no setup step.
    """
    path = path or os.environ.get("PICO_CHECKPOINT")
    if not path:
        repo = HF_MODEL_REPO.get(language)
        if not repo:
            raise SystemExit(
                f"no released weights for {language!r}: pass --checkpoint PATH "
                "or set PICO_CHECKPOINT"
            )
        return download_checkpoint(repo)
    if os.path.isdir(path):
        for name in DEFAULT_CKPT_NAMES:
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                return cand
        raise SystemExit(f"no {' / '.join(DEFAULT_CKPT_NAMES)} found in {path}")
    if not os.path.exists(path):
        # A repo id rather than a typo'd path: "org/name", no path separator
        # beyond the single slash and no extension.
        if "/" in path and not path.endswith(".pt"):
            return download_checkpoint(path)
        raise SystemExit(f"checkpoint not found: {path}")
    return path


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".mp4", ".aac")


def iter_audio_paths(inputs: Iterable[str], recursive: bool = True) -> list[str]:
    """Expand files and directories into a sorted list of audio paths."""
    out: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                out += [
                    os.path.join(root, f)
                    for f in files
                    if f.lower().endswith(AUDIO_EXTS)
                ]
                if not recursive:
                    break
        elif os.path.exists(item):
            out.append(item)
        else:
            raise SystemExit(f"no such file or directory: {item}")
    return sorted(out)


def write_records(records: list[dict], out_path: str | None, fmt: str) -> None:
    """Emit results as jsonl, json, csv or plain text."""
    fh = open(out_path, "w", encoding="utf-8") if out_path else sys.stdout
    try:
        if fmt == "jsonl":
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        elif fmt == "json":
            json.dump(records, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        elif fmt == "csv":
            import csv

            if not records:
                return
            cols = list(records[0])
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow(r)
        else:  # text
            for r in records:
                fh.write(f"{r.get('id', '')}\t{r.get('units', '')}\n")
    finally:
        if out_path:
            fh.close()
            print(f"wrote {len(records)} records -> {out_path}", file=sys.stderr)


def add_decode_args(ap) -> None:
    """Decode-time knobs, shared by every inference mode."""
    g = ap.add_argument_group("decoding")
    g.add_argument(
        "--smooth-frames",
        type=int,
        default=7,
        help="moving-average width over posteriors, in 10 ms frames. Frame "
        "predictions flicker between neighbouring units and each flicker "
        "would become a spurious unit (default: 7)",
    )
    g.add_argument(
        "--min-frames",
        type=int,
        default=4,
        help="drop any run shorter than this many frames (default: 4)",
    )
    g.add_argument(
        "--min-confidence", type=float, default=0.0, help="drop runs below this mean posterior"
    )
    g.add_argument(
        "--keep-silence", action="store_true", help="keep <sil> units in the output"
    )
    g.add_argument(
        "--split-long-runs",
        action="store_true",
        help="re-split a long run into repeated units to recover geminates. "
        "Off by default: it measured worse on real audio (UER 0.58 -> 0.85), "
        "because an uncertain model emits smeared runs that split into "
        "spurious repeats",
    )
    g.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")


def decode_kwargs(args) -> dict:
    return {
        "smooth_frames": args.smooth_frames,
        "min_frames": args.min_frames,
        "min_confidence": args.min_confidence,
        "drop_silence": not args.keep_silence,
        "split_long_runs": args.split_long_runs,
    }
