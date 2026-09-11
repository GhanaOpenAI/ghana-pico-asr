"""Publish a trained checkpoint to the Hugging Face Hub.

The Hub holds the weights; GitHub holds the code. Keeping binaries out of the
git repo is the obvious half. The less obvious half is that the model card
carries **no inference script** — it points at `ghana-pico-asr` instead, so
there is exactly one implementation of decoding. A copy pasted into a model
card drifts the first time a decode default changes, and then two users get
different transcripts from the same weights.

Fetching through `huggingface_hub` also makes usage countable, so the download
figure reflects real use rather than nothing.

    python scripts/publish_model.py --checkpoint path/to/best.pt \
        --repo-id ghanaopenai/ghana-pico-asr-twi
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEADER = """---
license: cc-by-nc-4.0
language: [tw, ak]
pipeline_tag: automatic-speech-recognition
library_name: ghana-pico-asr
tags: [audio, speech, twi, akan, ghana, phoneme-recognition, grapheme-units, 2d-cnn]
datasets:
  - ghanaopenai/twi-grapheme-unit-features
  - ghanaopenai/twi-grapheme-unit-pairs
metrics: [unit-error-rate]
---

"""

USAGE = """
## Using it

There is one implementation of inference, in the
[`ghana-pico-asr`](https://github.com/ghanaopenai/ghana-pico-asr) repository.
This card deliberately ships no copy of it: a decode default changed in one
place and not the other is how two people get different transcripts from the
same weights.

```bash
pip install git+https://github.com/ghanaopenai/ghana-pico-asr.git
```

The weights are fetched from this repo automatically — there is no download
step:

```bash
pico transcribe speech.wav                  # single file
pico hf-dataset org/dataset --limit 100     # straight from a HF dataset
pico web                                    # browser UI
```

```python
from ghana_pico_asr.cli._common import resolve_checkpoint
from ghana_pico_asr.infer import UnitTagger

tagger = UnitTagger(resolve_checkpoint(None))   # pulls from this repo
print(tagger.transcribe("speech.wav"))          # 'ɔ y ɛ n e h o'
```

To pin a specific checkpoint instead:

```bash
pico transcribe speech.wav --checkpoint {repo_id}
```

## What this outputs

Grapheme units, not words. Turning `ɔ y ɛ n e h o` into "Ɔyɛ ne ho" needs a
text-recovery model; the pairs for training one are published at
[`ghanaopenai/twi-grapheme-unit-pairs`](https://huggingface.co/datasets/ghanaopenai/twi-grapheme-unit-pairs).
"""


def build_card(repo_id: str, card_path: str) -> str:
    body = io.open(card_path, encoding="utf-8").read()
    # The repo's card opens with its own H1; keep it and graft the Hub header
    # and a usage section that points back at the code.
    marker = "## Intended use"
    if marker in body:
        body = body.replace(marker, USAGE.format(repo_id=repo_id) + "\n" + marker, 1)
    else:
        body += USAGE.format(repo_id=repo_id)
    return HEADER + body


def build_config(ckpt: dict) -> dict:
    tc = ckpt.get("train_config", {})
    return {
        "architecture": "PicoASRNet",
        "library": "ghana-pico-asr",
        "code": "https://github.com/ghanaopenai/ghana-pico-asr",
        "language": ckpt.get("language"),
        "n_classes": ckpt.get("n_classes"),
        "vocab": ckpt.get("vocab"),
        "receptive_field_ms": ckpt.get("receptive_field_ms"),
        "channels": list(tc.get("channels", ())),
        "temporal_dim": tc.get("temporal_dim"),
        "dilations": list(tc.get("dilations", ())),
        "feature_config": ckpt.get("feature_config"),
        "norm": ckpt.get("norm"),
        "chunk_config": ckpt.get("chunk_config"),
        "epoch": ckpt.get("epoch"),
        "select_metric": ckpt.get("select_metric"),
        # Decode defaults belong with the weights: they were tuned for this
        # model, and the 4M-parameter models wanted different ones.
        "decode": {"smooth_frames": 7, "min_frames": 4, "drop_silence": True},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", default="ghanaopenai/ghana-pico-asr-twi")
    ap.add_argument("--card", default="MODEL_CARD.md")
    ap.add_argument("--metrics", default=None, help="JSON file of eval results")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    import torch
    from huggingface_hub import HfApi

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = build_config(ckpt)
    card = build_card(args.repo_id, args.card)

    print(f"checkpoint : {args.checkpoint} (epoch {ckpt.get('epoch')})")
    print(f"classes    : {cfg['n_classes']}  context {cfg['receptive_field_ms']} ms")
    print(f"card       : {len(card):,} chars")
    if args.dry_run:
        print("\n--- card header ---")
        print("\n".join(card.split("\n")[:18]))
        return 0

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=args.checkpoint, path_in_repo="best.pt",
                    repo_id=args.repo_id, commit_message="model weights")
    api.upload_file(path_or_fileobj=json.dumps(cfg, ensure_ascii=False, indent=2).encode(),
                    path_in_repo="config.json", repo_id=args.repo_id,
                    commit_message="architecture, features and decode defaults")
    if args.metrics and os.path.exists(args.metrics):
        api.upload_file(path_or_fileobj=args.metrics, path_in_repo="metrics.json",
                        repo_id=args.repo_id, commit_message="evaluation results")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo_id, commit_message="model card")
    print(f"\npublished -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
