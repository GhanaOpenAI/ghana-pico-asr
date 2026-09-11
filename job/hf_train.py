"""Training entrypoint for Hugging Face Jobs.

Modal gave the trainer one writable volume holding everything. A Job instead
gets a read-only dataset mount, a read-write bucket, and a container disk that
dies with the job, so this assembles the `/data` root the trainer expects out
of those three:

    /data/features/mel       -> the dataset mount (read-only, 24 GB)
    /data/features/manifest  -> the dataset mount
    /data/features/vocab.json-> the dataset mount
    /data/features/index       container disk: a rebuildable cache, and the
                               build writes 300+ small files, which belongs on
                               local disk rather than a network bucket
    /data/checkpoints        -> the bucket, so `last.pt` survives the job and
                               a restarted job resumes instead of starting over

Run it through `scripts/launch_hf_job.py`, which passes the same flags
`scripts/spawn_training.py` takes for Modal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

DATA = "/data"
STORE = os.environ.get("PICO_STORE", "/store")     # read-only dataset mount
WORK = os.environ.get("PICO_WORK", "/work")        # read-write bucket mount
LOCAL = os.environ.get("PICO_LOCAL", "/scratch")   # container disk

FEATURE_PARTS = ("mel", "manifest")


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001 - diagnostics must never abort a 30 h run
        return ""


def report_environment() -> None:
    print("[env] python", sys.version.split()[0], flush=True)
    try:
        import torch

        print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"[env] gpu {p.name} {p.total_memory / 1e9:.0f} GB sm{p.major}{p.minor}",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[env] torch unavailable:", exc, flush=True)
    print("[env] cpus", os.cpu_count(), flush=True)
    # LOCAL may not exist yet, so report the filesystem that will hold it --
    # the 24 GB stage needs headroom and a surprise here is worth seeing first.
    os.makedirs(LOCAL, exist_ok=True)
    disk = _sh(["df", "-h", LOCAL]).splitlines()
    print("[env] disk", disk[-1] if disk else "unavailable", flush=True)


def stage_features(copy_local: bool) -> str:
    """Point `/data/features` at the mounted store, optionally via local disk.

    Training reads mel blobs at random across millions of chunks. That is the
    worst case for a network-backed mount, so by default the store is copied to
    the container disk first -- one sequential read, then every training read is
    local. `--no-copy` trains straight off the mount if the mount is fast.
    """
    src = os.path.join(STORE, "features")
    if not os.path.isdir(src):
        raise SystemExit(f"dataset mount missing: {src} (is -v hf://datasets/... set?)")

    feat = os.path.join(DATA, "features")
    os.makedirs(feat, exist_ok=True)

    if copy_local:
        dest_root = os.path.join(LOCAL, "features")
        os.makedirs(dest_root, exist_ok=True)
        for part in FEATURE_PARTS:
            s, d = os.path.join(src, part), os.path.join(dest_root, part)
            if os.path.isdir(d) and os.listdir(d):
                print(f"[stage] {part}: already local, skipping", flush=True)
                continue
            t0 = time.time()
            shutil.copytree(s, d, dirs_exist_ok=True)
            n = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
            dt = max(time.time() - t0, 1e-6)
            print(f"[stage] {part}: {n / 1e9:.2f} GB in {dt:.0f}s "
                  f"({n / dt / 1e6:.0f} MB/s)", flush=True)
        origin = dest_root
    else:
        origin = src

    for part in FEATURE_PARTS:
        _link(os.path.join(origin, part), os.path.join(feat, part))
    # Copied rather than linked: the trainer refreshes this file, and a link
    # would aim that write at the read-only dataset mount.
    shutil.copyfile(os.path.join(src, "vocab.json"),
                    os.path.join(feat, "vocab.json"))

    # Written during training, so it cannot live on the read-only mount.
    os.makedirs(os.path.join(LOCAL, "index"), exist_ok=True)
    _link(os.path.join(LOCAL, "index"), os.path.join(feat, "index"))
    return feat


def _link(src: str, dst: str) -> None:
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            return
    os.symlink(src, dst)


def stage_checkpoints(run_name: str, fresh: bool) -> str:
    """Put checkpoints on the bucket so a restarted job resumes."""
    ck_root = os.path.join(WORK, "checkpoints")
    os.makedirs(ck_root, exist_ok=True)
    _link(ck_root, os.path.join(DATA, "checkpoints"))

    run_dir = os.path.join(ck_root, run_name)
    last = os.path.join(run_dir, "last.pt")
    if fresh and os.path.exists(last):
        # A "fresh" run must not silently inherit a previous run's optimiser
        # state just because the name matches.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.move(run_dir, f"{run_dir}.superseded-{stamp}")
        print(f"[ckpt] existing {run_name} moved aside -> {run_name}.superseded-{stamp}",
              flush=True)
    elif os.path.exists(last):
        print(f"[ckpt] found {last}; the trainer will resume from it", flush=True)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="hf30")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--channels", default="96,192,384")
    ap.add_argument("--temporal-dim", type=int, default=640)
    ap.add_argument("--dilations", default="1,2,4,8,16")
    ap.add_argument("--max-chunks-per-split", type=int, default=1_200_000)
    ap.add_argument("--splits", default="tts,asr,kuma,female,agric,bible")
    ap.add_argument("--select-metric", default="balanced_acc")
    ap.add_argument("--max-train-chunks", type=int, default=0,
                    help="subset the train split; used to smoke-test a flavor")
    ap.add_argument("--no-copy", action="store_true",
                    help="train straight off the dataset mount, without staging locally")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint for this run name")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report_environment()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stage_features(copy_local=not args.no_copy)
    stage_checkpoints(args.run_name, fresh=args.fresh)

    from ghana_pico_asr import config as C
    from ghana_pico_asr.trainer import train

    ccfg = C.ChunkConfig(
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        max_chunks_per_split=args.max_chunks_per_split,
    )
    tcfg = C.TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        channels=tuple(int(c) for c in args.channels.split(",") if c.strip()),
        temporal_dim=args.temporal_dim,
        dilations=tuple(int(d) for d in args.dilations.split(",") if d.strip()),
        run_name=args.run_name,
        select_metric=args.select_metric,
        max_train_chunks=args.max_train_chunks,
    )

    summary = train(DATA, ccfg, tcfg, device="cuda")
    slim = {k: v for k, v in summary.items()
            if k not in ("test_per_class", "units", "history", "test_contrasts",
                         "provenance")}
    print("[done] " + json.dumps(slim, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
