"""Start a training run that outlives this machine.

`modal run` ties the run's lifetime to a client this computer owns, and even
`--detach` has been unreliable in practice. `spawn` instead queues the call
server-side and returns immediately with a call id, so the run continues if
this computer sleeps, loses network, or is killed outright. Combined with the
trainer's per-epoch `last.pt`, a run then survives almost anything.

    # once, and after any code change:
    modal deploy modal_app/train.py

    # then, per run:
    python scripts/spawn_training.py --run-name xl --epochs 8 \
        --channels 96,192,384 --temporal-dim 640 --dilations 1,2,4,8,16

    # check on it later, from anywhere:
    python scripts/spawn_training.py --status <call-id>
    python scripts/spawn_training.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_NAME = "twi-phoneme-2dcnn"
RUNS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".spawned_runs.json"
)


def _load_runs() -> dict:
    if not os.path.exists(RUNS_FILE):
        return {}
    try:
        with open(RUNS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 - a corrupt record must not block a run
        return {}


def _record(run_name: str, call_id: str, payload: dict) -> None:
    """Keep call ids on disk so a run can be found again after a reboot."""
    runs = _load_runs()
    runs[run_name] = {"call_id": call_id, **payload}
    with open(RUNS_FILE, "w", encoding="utf-8") as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)
    print(f"recorded -> {RUNS_FILE}")


def _param_count(channels, temporal_dim, dilations) -> int:
    from ghana_pico_asr.model import PicoASRNet

    return PicoASRNet(
        n_classes=37,
        channels=tuple(channels),
        temporal_dim=temporal_dim,
        dilations=tuple(dilations),
    ).n_params()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--status", default=None, metavar="CALL_ID", help="check a spawned run and exit"
    )
    ap.add_argument("--list", action="store_true", help="list recorded runs and exit")
    ap.add_argument("--run-name", default="xl")
    ap.add_argument("--epochs", type=int, default=8)
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
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import modal

    if args.list:
        runs = _load_runs()
        if not runs:
            print("no recorded runs")
            return 0
        for name, r in runs.items():
            print(f"{name:<12} {r['call_id']}")
        return 0

    if args.status:
        call = modal.FunctionCall.from_id(args.status)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            print(f"call {args.status}: still running")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"call {args.status}: FAILED - {type(exc).__name__}: {exc}")
            return 1
        if isinstance(result, dict):
            slim = {k: v for k, v in result.items() if k != "test_per_class"}
            print(json.dumps(slim, ensure_ascii=False, indent=2))
        else:
            print(result)
        return 0

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    channels = [int(c) for c in args.channels.split(",") if c.strip()]
    dilations = [int(d) for d in args.dilations.split(",") if d.strip()]

    ccfg = {"splits": splits, "max_chunks_per_split": args.max_chunks_per_split}
    tcfg = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "channels": tuple(channels),
        "temporal_dim": args.temporal_dim,
        "dilations": tuple(dilations),
        "run_name": args.run_name,
        "select_metric": args.select_metric,
    }

    try:
        fn = modal.Function.from_name(APP_NAME, "run_training")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"could not find `run_training` in deployed app {APP_NAME!r}: {exc}\n"
            "  Deploy it first:  modal deploy modal_app/train.py"
        ) from None

    call = fn.spawn(ccfg, tcfg)
    print(f"spawned {args.run_name!r}: call id {call.object_id}")
    print(f"  ~{_param_count(channels, args.temporal_dim, dilations):,} params")
    print(f"  splits {','.join(splits)}  cap {args.max_chunks_per_split:,} chunks/source")
    print(f"  epochs {args.epochs}, patience {args.patience}, select {args.select_metric}")
    print("\nThis run no longer depends on this machine. Check it with:")
    print(f"  python scripts/spawn_training.py --status {call.object_id}")
    print(f"  modal app logs {APP_NAME}")

    _record(
        args.run_name,
        call.object_id,
        {
            "ccfg": {**ccfg, "splits": list(splits)},
            "tcfg": {**tcfg, "channels": channels, "dilations": dilations},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
