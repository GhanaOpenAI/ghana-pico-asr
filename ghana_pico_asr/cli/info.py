"""Show what a checkpoint is and how it was made."""

from __future__ import annotations

from ._common import resolve_checkpoint


def add_args(ap) -> None:
    ap.add_argument("checkpoint", help="checkpoint file or directory")
    ap.add_argument("--json", action="store_true", help="print the provenance block as JSON")
    ap.add_argument("--vocab", action="store_true", help="list the unit inventory")


def run(args) -> int:
    import torch

    from .. import provenance as P

    path = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if args.json:
        print(P.to_json(ckpt))
        return 0

    print(f"checkpoint     {path}")
    print(P.describe(ckpt))
    if args.vocab:
        vocab = ckpt.get("vocab", [])
        print(f"\nvocabulary ({len(vocab)} classes, in class order):")
        for i in range(0, len(vocab), 12):
            print("  " + "  ".join(f"{j:>2}:{vocab[j]}" for j in range(i, min(i + 12, len(vocab)))))
    if not ckpt.get("provenance"):
        print(
            "\nnote: this checkpoint predates provenance recording, so corpus and\n"
            "      git details are unavailable. Architecture, vocabulary, feature\n"
            "      config and normalisation are present, so it can still be used\n"
            "      for inference and fine-tuning."
        )
    return 0
