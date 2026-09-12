"""Start the NLLB text-recovery fine-tune on Hugging Face Jobs.

Stage 2 of the pipeline. The parallel for stage 1 is
`scripts/launch_hf_job.py`; this takes the same shape — detached by default,
billed to an org, checkpoints on a bucket so a restarted job resumes.

It needs no dataset mount: the pairs are a 47 MB parquet pulled straight from
the Hub, unlike the 24 GB feature store stage 1 has to stage to local disk.

    python scripts/launch_recovery_job.py --smoke           # ~15 min, validates
    python scripts/launch_recovery_job.py --epochs 3
    hf jobs logs <job-id> --namespace ghananlpcommunity
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
DEFAULT_BUCKET = "ghananlpcommunity/pico-asr-runs"
# Torch 2.10 to match the pinned peft/transformers, not stage 1's 2.5.1:
# peft 0.19 reaches for `torch.float8_e8m0fnu`, which 2.5.1 does not have.
# Pinning libraries without pinning the runtime underneath them is only half a
# pin. `huggingface/transformers-pytorch-gpu` was the other candidate and ships
# transformers already, but has no `python` on PATH — only `python3` — so the
# container never starts.
DEFAULT_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"


def _stage_code() -> str:
    stage = os.path.join(tempfile.gettempdir(), "pico_recovery_job_code")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    for part in ("ghana_pico_asr", "job"):
        shutil.copytree(
            os.path.join(HERE, part), os.path.join(stage, part),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return stage


def _record(run_name: str, job_id: str, payload: dict) -> None:
    runs = {}
    if os.path.exists(RUNS_FILE):
        try:
            with open(RUNS_FILE, encoding="utf-8") as fh:
                runs = json.load(fh)
        except Exception:  # noqa: BLE001
            runs = {}
    runs[run_name] = {"job_id": job_id, **payload}
    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-name", default="nllb-recovery")
    ap.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--lang-code", default="twi_Latn")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--full-finetune", action="store_true")
    ap.add_argument("--max-uer", type=float, default=0.5)
    ap.add_argument("--machine-ratio", type=float, default=0.5)
    ap.add_argument("--spaced-input", action="store_true")
    ap.add_argument("--push-to", default=None)

    ap.add_argument("--flavor", default="l40sx1")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--timeout", default="24h")
    ap.add_argument("--smoke", action="store_true",
                    help="4k pairs, a third of an epoch — validates the path")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    run_name = f"{args.run_name}-smoke" if args.smoke else args.run_name
    train_args = [
        "--run-name", run_name,
        "--base-model", args.base_model,
        "--lang-code", args.lang_code,
        "--epochs", str(0.34 if args.smoke else args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--max-uer", str(args.max_uer),
        "--machine-ratio", str(args.machine_ratio),
    ]
    if args.full_finetune:
        train_args.append("--full-finetune")
    if args.spaced_input:
        train_args.append("--spaced-input")
    if args.smoke:
        train_args += ["--limit", "4000", "--eval-n", "200"]
    if args.push_to and not args.smoke:
        train_args += ["--push-to", args.push_to]

    cmd = [
        "hf", "jobs", "run", "--detach",
        "--flavor", args.flavor,
        "--namespace", args.namespace,
        "--timeout", args.timeout,
        "--secrets", "HF_TOKEN",
        "-v", f"hf://buckets/{args.bucket}:/work:rw",
        "-v", f"{_stage_code()}:/code:ro",
        "-e", "PICO_WORK=/work",
        "-e", "HF_HUB_ENABLE_HF_TRANSFER=1",
        "-e", "PYTHONUNBUFFERED=1",
        args.image,
        # A plain argv, no shell: the entrypoint installs what the image lacks
        # itself. Routing `pip install ... && python ...` through `sh -c` puts
        # the command through two layers of quoting, where `peft>=0.11` reads
        # as a redirection and the command string can be taken for a filename.
        "python", "/code/job/hf_recovery.py", *train_args,
    ]

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    print(f"launching {run_name!r} on {args.flavor} (billed to {args.namespace})")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    out = proc.stdout.strip()
    print(out)
    if proc.returncode != 0:
        return proc.returncode

    job_id = next((t[3:] for t in out.split() if t.startswith("id=")), "")
    if job_id:
        _record(run_name, job_id, {"flavor": args.flavor, "namespace": args.namespace,
                                   "base_model": args.base_model, "smoke": args.smoke})
        print(f"\n  hf jobs logs {job_id} --namespace {args.namespace}")
        print(f"  hf jobs cancel {job_id} --namespace {args.namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
