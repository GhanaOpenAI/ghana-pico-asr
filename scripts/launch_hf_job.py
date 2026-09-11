"""Start a training run on Hugging Face Jobs.

The Modal equivalent is `scripts/spawn_training.py`; this takes the same
training flags and adds the ones a Job needs (hardware flavor, billing
namespace). `--detach` is the default, so the run outlives this machine exactly
as `spawn` does on Modal.

    # smoke-test a flavor end to end, ~10 min
    python scripts/launch_hf_job.py --smoke --flavor l40sx1

    # the real thing
    python scripts/launch_hf_job.py --run-name hf30 --epochs 30 --patience 0

    hf jobs logs <job-id> --namespace ghananlpcommunity
    hf jobs cancel <job-id> --namespace ghananlpcommunity

Storage, since a Job has no single writable volume like Modal's:

* the feature store is mounted read-only from the published dataset
* checkpoints go to a read-write bucket, so a restarted job resumes
* the label index is rebuilt on the container disk (deterministically, so the
  split is identical either way)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_FILE = os.path.join(HERE, ".hf_jobs.json")

DEFAULT_NAMESPACE = "ghananlpcommunity"
DEFAULT_DATASET = "ghanaopenai/twi-grapheme-unit-features"
DEFAULT_BUCKET = "ghananlpcommunity/pico-asr-runs"
# Torch preinstalled against CUDA 12.1, matching the Modal image; the training
# path needs nothing else (the aligner is imported lazily and never runs here).
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"


def _stage_code() -> str:
    """Copy the package into a path the mount spec can carry.

    `-v LOCAL:/code` splits on ':' and the checkout lives under a directory with
    a space in its name, so the source is staged somewhere plain first. Only
    what training imports is copied -- tests and Modal glue are not needed in
    the container.
    """
    stage = os.path.join(tempfile.gettempdir(), "pico_asr_job_code")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for part in ("ghana_pico_asr", "job"):
        shutil.copytree(
            os.path.join(HERE, part),
            os.path.join(stage, part),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return stage


def _record(run_name: str, job_id: str, payload: dict) -> None:
    runs = {}
    if os.path.exists(RUNS_FILE):
        try:
            with open(RUNS_FILE, encoding="utf-8") as fh:
                runs = json.load(fh)
        except Exception:  # noqa: BLE001 - a corrupt record must not block a run
            runs = {}
    runs[run_name] = {"job_id": job_id, **payload}
    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-name", default="hf30")
    ap.add_argument("--epochs", type=int, default=30)
    # 0 by default: the cosine schedule is sized to the full budget, so a run
    # stopped at 20 of 30 sits at ~25% of peak LR rather than annealed, and a
    # shorter properly-annealed run usually beats it. Watch the metrics and
    # cancel by hand instead.
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--channels", default="96,192,384")
    ap.add_argument("--temporal-dim", type=int, default=640)
    ap.add_argument("--dilations", default="1,2,4,8,16")
    ap.add_argument("--max-chunks-per-split", type=int, default=1_200_000)
    ap.add_argument("--splits", default="tts,asr,kuma,female,agric,bible")
    ap.add_argument("--select-metric", default="balanced_acc")
    ap.add_argument("--fresh", action="store_true", default=True,
                    help="ignore any checkpoint already under this run name")
    ap.add_argument("--resume", dest="fresh", action="store_false",
                    help="continue an interrupted job from its bucket checkpoint")

    ap.add_argument("--flavor", default="a10g-large")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--timeout", default="48h")
    ap.add_argument("--no-copy", action="store_true",
                    help="train off the dataset mount instead of staging to local disk")
    ap.add_argument("--smoke", action="store_true",
                    help="short run on 60k train chunks, to validate a flavor")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    run_name = f"{args.run_name}-smoke" if args.smoke else args.run_name
    epochs = 1 if args.smoke else args.epochs

    train_args = [
        "--run-name", run_name,
        "--epochs", str(epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num-workers", str(args.num_workers),
        "--channels", args.channels,
        "--temporal-dim", str(args.temporal_dim),
        "--dilations", args.dilations,
        "--max-chunks-per-split", str(args.max_chunks_per_split),
        "--splits", args.splits,
        "--select-metric", args.select_metric,
    ]
    if args.fresh:
        train_args.append("--fresh")
    if args.no_copy:
        train_args.append("--no-copy")
    if args.smoke:
        train_args += ["--max-train-chunks", "60000"]

    cmd = [
        "hf", "jobs", "run", "--detach",
        "--flavor", args.flavor,
        "--namespace", args.namespace,
        "--timeout", args.timeout,
        "--secrets", "HF_TOKEN",
        "-v", f"hf://datasets/{args.dataset}:/store:ro",
        "-v", f"hf://buckets/{args.bucket}:/work:rw",
        "-v", f"{_stage_code()}:/code:ro",
        "-e", "PICO_STORE=/store",
        "-e", "PICO_WORK=/work",
        "-e", "PICO_LOCAL=/scratch",
        "-e", "HF_HUB_ENABLE_HF_TRANSFER=1",
        "-e", "PYTHONUNBUFFERED=1",
        args.image,
        "python", "/code/job/hf_train.py", *train_args,
    ]

    if args.dry_run:
        print(" \\\n  ".join(cmd))
        return 0

    print(f"launching {run_name!r} on {args.flavor} "
          f"(billed to {args.namespace})", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    out = proc.stdout.strip()
    print(out)
    if proc.returncode != 0:
        return proc.returncode

    # Output is `id=<id> url=<url>`; the last token is the URL, not the id.
    job_id = next(
        (tok[len("id="):] for tok in out.split() if tok.startswith("id=")), ""
    )
    if job_id:
        _record(run_name, job_id, {
            "flavor": args.flavor,
            "namespace": args.namespace,
            "epochs": args.epochs,
            "patience": args.patience,
            "smoke": args.smoke,
        })
        print(f"\n  hf jobs logs {job_id} --namespace {args.namespace}")
        print(f"  hf jobs cancel {job_id} --namespace {args.namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
