"""Fine-tune a published checkpoint on new data.

Fine-tuning differs from training in three ways that matter:

* **Weights come from a checkpoint, optimiser state does not.** A fresh LR
  schedule is what you want; continuing ours would start you at its annealed
  tail.
* **The vocabulary may differ.** New data can contain units the original never
  saw, or lack ones it did. A plain ``load_state_dict`` would silently
  mis-assign every class, so the output layer is remapped by unit *name*:
  shared units keep their trained weights, new units start fresh.
* **Feature extraction must match exactly.** Mel parameters and the
  normalisation statistics are read from the checkpoint, not from local config,
  because features computed differently make the weights meaningless.
"""

from __future__ import annotations

import torch

from . import config as C
from .languages import get_language
from .model import PicoASRNet


class FinetuneError(RuntimeError):
    pass


def load_for_finetune(
    checkpoint: str,
    new_vocab: list[str],
    device: str = "cuda",
    freeze_trunk: bool = False,
    reset_head: bool = False,
) -> tuple[PicoASRNet, dict]:
    """Build a model initialised from ``checkpoint`` for ``new_vocab``.

    Returns the model and a report describing what was transferred, so a
    fine-tune run records how much of it was actually inherited.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tc = ckpt.get("train_config") or {}
    old_vocab: list[str] = ckpt["vocab"]

    feat = ckpt.get("feature_config") or {}
    mismatched = {
        k: (feat.get(k), getattr(C, k.upper()))
        for k in ("sample_rate", "n_fft", "hop_length", "n_mels")
        if k in feat and feat[k] != getattr(C, k.upper(), feat[k])
    }
    if mismatched:
        raise FinetuneError(
            "feature config differs from this checkpoint, so its weights would be "
            f"meaningless: {mismatched}. Align config.py with the checkpoint, or "
            "train from scratch."
        )

    model = PicoASRNet(
        n_classes=len(new_vocab),
        channels=tuple(tc.get("channels", (48, 96, 192))),
        temporal_dim=tc.get("temporal_dim", 384),
        dilations=tuple(tc.get("dilations", C.TEMPORAL_DILATIONS)),
        dropout=tc.get("dropout", 0.1),
    )

    state = dict(ckpt["model"])
    # The final 1x1 conv is the only vocabulary-dependent tensor.
    head_w = "head.1.weight"
    head_b = "head.1.bias"
    transferred, fresh = [], []
    if not reset_head and head_w in state and len(old_vocab) == state[head_w].shape[0]:
        old_idx = {u: i for i, u in enumerate(old_vocab)}
        new_w = model.state_dict()[head_w].clone()
        new_b = model.state_dict()[head_b].clone()
        for j, unit in enumerate(new_vocab):
            i = old_idx.get(unit)
            if i is None:
                fresh.append(unit)
            else:
                new_w[j] = state[head_w][i]
                new_b[j] = state[head_b][i]
                transferred.append(unit)
        state[head_w], state[head_b] = new_w, new_b
    else:
        fresh = list(new_vocab)
        state.pop(head_w, None)
        state.pop(head_b, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)

    if freeze_trunk:
        for name, param in model.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = False

    report = {
        "from_checkpoint": checkpoint,
        "source_language": ckpt.get("language"),
        "source_epoch": ckpt.get("epoch"),
        "source_val": {k: v for k, v in (ckpt.get("val") or {}).items() if isinstance(v, float)},
        "old_vocab_size": len(old_vocab),
        "new_vocab_size": len(new_vocab),
        "units_transferred": len(transferred),
        "units_new": fresh,
        "units_dropped": sorted(set(old_vocab) - set(new_vocab)),
        "head_reset": bool(reset_head),
        "trunk_frozen": bool(freeze_trunk),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "norm": ckpt.get("norm"),
    }
    if missing and not reset_head:
        raise FinetuneError(
            f"checkpoint is missing weights this architecture needs: {list(missing)[:6]}. "
            "Its train_config may not describe the weights it holds."
        )
    return model, report


def describe(report: dict) -> str:
    lines = [
        f"initialised from {report['from_checkpoint']}",
        f"  source          {report.get('source_language')} @ epoch {report.get('source_epoch')}",
        f"  vocabulary      {report['old_vocab_size']} -> {report['new_vocab_size']}",
        f"  transferred     {report['units_transferred']} unit classes kept their weights",
    ]
    if report["units_new"]:
        lines.append(
            f"  new units       {len(report['units_new'])} start fresh: "
            + " ".join(report["units_new"][:20])
        )
    if report["units_dropped"]:
        lines.append(
            f"  dropped units   {len(report['units_dropped'])}: "
            + " ".join(report["units_dropped"][:20])
        )
    if report["trunk_frozen"]:
        lines.append("  trunk frozen    only the output layer will train")
    return "\n".join(lines)
