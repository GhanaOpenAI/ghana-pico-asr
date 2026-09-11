"""Fine-tune a checkpoint on a locally prepared feature store."""

from __future__ import annotations

import json

from ._common import resolve_checkpoint


def add_args(ap) -> None:
    ap.add_argument("-c", "--checkpoint", required=True, help="checkpoint to start from")
    ap.add_argument(
        "--data-root",
        required=True,
        help="feature store directory (built by the prepare stage)",
    )
    ap.add_argument("-l", "--language", default=None, help="language code (default: from data)")
    ap.add_argument("--splits", default="tts,asr,kuma", help="which prepared sources to use")
    ap.add_argument("--run-name", default="finetune")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="fine-tune LR, deliberately ~6x lower than the 3e-4 used from "
        "scratch (default: 5e-5)",
    )
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument(
        "--freeze-trunk",
        action="store_true",
        help="train only the output layer — for a small dataset or a new vocabulary",
    )
    ap.add_argument(
        "--reset-head",
        action="store_true",
        help="discard the output layer entirely instead of remapping it by unit name",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="report the transfer plan and exit")


def run(args) -> int:
    from .. import config as C
    from .. import dataset as D
    from ..finetune import describe, load_for_finetune
    from ..languages import get_language
    from ..trainer import train

    ckpt = resolve_checkpoint(args.checkpoint)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    ccfg = C.ChunkConfig(splits=splits)

    # The new vocabulary comes from the new data, not the checkpoint.
    vocab = D.load_or_build_vocab(args.data_root, ccfg)
    lang = get_language(args.language)

    model, report = load_for_finetune(
        ckpt,
        vocab["units"],
        device="cpu" if args.dry_run else args.device,
        freeze_trunk=args.freeze_trunk,
        reset_head=args.reset_head,
    )
    print(describe(report))

    if args.dry_run:
        print("\n--dry-run: nothing trained.")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    tcfg = C.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.num_workers,
        run_name=args.run_name,
    )
    summary = train(
        args.data_root, ccfg, tcfg, device=args.device, lang_code=lang.code, init_model=model
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "test_per_class"},
                     ensure_ascii=False, indent=2))
    return 0
