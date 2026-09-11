"""Training loop for the frame-wise 2D CNN, plus evaluation.

The loss is cross-entropy over every 10 ms frame, ignoring frames marked
``IGNORE_INDEX`` (uncertain alignment, out-of-vocabulary units, time-masked
regions).

Two evaluation views, because they answer different questions:

* **frame accuracy / macro-F1** — how well the model names each 10 ms frame;
* **unit error rate** — the edit distance between the collapsed prediction and
  the collapsed reference. This is the number that matters for the actual task,
  since a run of frames collapses into one unit at inference.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as D
from . import provenance as P
from .languages import get_language
from .model import PicoASRNet

def class_weights(counts: np.ndarray, alpha: float) -> torch.Tensor:
    """Tempered inverse-frequency weights, normalised to mean 1."""
    c = np.maximum(counts.astype(np.float64), 1.0)
    w = (c.sum() / c) ** alpha
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def collapse(seq: np.ndarray, sil_class: int = 0) -> list[int]:
    """Collapse a frame label run into a unit sequence, dropping silence."""
    out: list[int] = []
    prev = -1
    for v in seq:
        if v != prev:
            if v != sil_class and v >= 0:
                out.append(int(v))
            prev = v
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj))
        prev = cur
    return prev[-1]


@torch.inference_mode()
def evaluate(model, loader, device, n_classes: int) -> dict:
    model.eval()
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long, device=device)
    total_loss = 0.0
    n_frames = 0
    errors = 0
    ref_len = 0
    lossf = nn.CrossEntropyLoss(reduction="sum", ignore_index=C.IGNORE_INDEX)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        logits = logits.float()
        total_loss += float(lossf(logits, y))

        pred = logits.argmax(1)
        mask = y != C.IGNORE_INDEX
        conf.index_put_((y[mask], pred[mask]), torch.ones_like(y[mask]), accumulate=True)
        n_frames += int(mask.sum())

        # Unit error rate, per chunk, over the collapsed sequences.
        pred_c = pred.cpu().numpy()
        y_c = y.cpu().numpy()
        for i in range(y_c.shape[0]):
            keep = y_c[i] != C.IGNORE_INDEX
            if not keep.any():
                continue
            ref = collapse(y_c[i][keep])
            hyp = collapse(pred_c[i][keep])
            errors += edit_distance(hyp, ref)
            ref_len += len(ref)

    conf = conf.cpu().numpy()
    support = conf.sum(1)
    pred_total = conf.sum(0)
    recall = np.divide(np.diag(conf), support, out=np.zeros(n_classes), where=support > 0)
    prec = np.divide(np.diag(conf), pred_total, out=np.zeros(n_classes), where=pred_total > 0)
    f1 = np.divide(
        2 * prec * recall, prec + recall, out=np.zeros(n_classes), where=(prec + recall) > 0
    )
    seen = support > 0
    return {
        "loss": total_loss / max(n_frames, 1),
        "frame_acc": float(np.trace(conf) / max(conf.sum(), 1)),
        "balanced_acc": float(recall[seen].mean()),
        "macro_f1": float(f1[seen].mean()),
        "unit_error_rate": float(errors / max(ref_len, 1)),
        "n_frames": int(n_frames),
        "n_ref_units": int(ref_len),
        "confusion": conf,
        "recall": recall,
        "support": support,
    }


def contrast_report(metrics: dict, units: list[str], lang=None) -> dict:
    """Per-pair recall and cross-confusion for the language's contrast pairs."""
    lang = lang or get_language()
    conf = metrics["confusion"]
    idx = {u: i for i, u in enumerate(units)}
    out = {}
    for a, b in lang.contrasts:
        if a not in idx or b not in idx:
            continue
        ia, ib = idx[a], idx[b]
        sa, sb = conf[ia].sum(), conf[ib].sum()
        out[f"{a}/{b}"] = {
            f"recall_{a}": round(float(conf[ia, ia] / sa), 4) if sa else None,
            f"recall_{b}": round(float(conf[ib, ib] / sb), 4) if sb else None,
            f"{a}_as_{b}": round(float(conf[ia, ib] / sa), 4) if sa else None,
            f"{b}_as_{a}": round(float(conf[ib, ia] / sb), 4) if sb else None,
            "support_frames": [int(sa), int(sb)],
        }
    return out


def train(
    root: str,
    ccfg: C.ChunkConfig | None = None,
    tcfg: C.TrainConfig | None = None,
    device: str = "cuda",
    lang_code: str = C.LANGUAGE,
    init_model=None,
) -> dict:
    ccfg = ccfg or C.ChunkConfig()
    tcfg = tcfg or C.TrainConfig()
    lang = get_language(lang_code)
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("[data] building vocabulary...", flush=True)
    vocab = D.load_or_build_vocab(root, ccfg)
    units = vocab["units"]
    n_classes = len(units)
    print(f"[data] {n_classes} classes: {' '.join(units)}", flush=True)

    print("[data] building frame label tracks...", flush=True)
    t0 = time.time()
    index = D.load_or_build_index(root, ccfg, vocab)
    st = index["stats"]
    labelled = st.get("labelled_frames", 0)
    frames = max(st.get("frames", 1), 1)
    print(
        f"[data] {len(index['frame_start']):,} chunks over {st.get('utts', 0):,} utts "
        f"in {time.time() - t0:.0f}s | {labelled:,}/{frames:,} frames labelled "
        f"({100 * labelled / frames:.1f}%)",
        flush=True,
    )

    mean, std = D.load_or_estimate_norm(root, index, ccfg)
    print(f"[data] log-mel norm: mean={mean:.4f} std={std:.4f}", flush=True)

    subsets = D.split_subsets(index, tcfg)
    if tcfg.max_train_chunks and len(subsets["train"]) > tcfg.max_train_chunks:
        rng = np.random.default_rng(tcfg.seed)
        subsets["train"] = np.sort(
            rng.choice(subsets["train"], size=tcfg.max_train_chunks, replace=False)
        )
    for name, sel in subsets.items():
        print(f"[data] {name}: {len(sel):,} chunks", flush=True)

    ds = {
        name: D.ChunkDataset(
            index,
            subset=sel,
            mean=mean,
            std=std,
            augment=(name == "train"),
            train_cfg=tcfg,
        )
        for name, sel in subsets.items()
    }
    loaders = {
        name: DataLoader(
            d,
            batch_size=tcfg.batch_size,
            shuffle=(name == "train"),
            num_workers=tcfg.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=(name == "train"),
            persistent_workers=tcfg.num_workers > 0,
            prefetch_factor=4 if tcfg.num_workers > 0 else None,
        )
        for name, d in ds.items()
    }

    # Fine-tuning passes a model already initialised from a checkpoint and
    # remapped to this vocabulary; training from scratch builds a fresh one.
    if init_model is not None:
        model = init_model.to(device)
        if model.n_classes != n_classes:
            raise ValueError(
                f"init_model has {model.n_classes} classes but this data needs {n_classes}"
            )
    else:
        model = PicoASRNet(
            n_classes=n_classes,
            channels=tuple(tcfg.channels),
            temporal_dim=tcfg.temporal_dim,
            dilations=tuple(tcfg.dilations),
            dropout=tcfg.dropout,
        ).to(device)
    print(
        f"[model] {model.n_params():,} params | receptive field "
        f"{model.receptive_field()} frames = {model.receptive_field_ms()} ms",
        flush=True,
    )

    prov = P.build(root, ccfg, tcfg, lang.code)
    print("[data] counting frame classes for weighting...", flush=True)
    counts = D.frame_class_counts(index, subsets["train"], n_classes)
    median_frames = D.unit_median_frames(root, ccfg, vocab)
    weights = class_weights(counts, tcfg.class_weight_alpha).to(device)
    lossf = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=tcfg.label_smoothing,
        ignore_index=C.IGNORE_INDEX,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    steps_per_epoch = max(1, len(loaders["train"]))
    total_steps = steps_per_epoch * tcfg.epochs
    warmup = max(1, int(tcfg.warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    ckpt_dir = os.path.join(root, C.CKPT_DIR, tcfg.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume support. A preempted container is restarted with the same input,
    # so without this a long run silently begins again from scratch — we lost
    # 5 of 6 epochs that way. `last.pt` carries optimiser and scheduler state
    # so the cosine schedule continues rather than restarting its warmup.
    resume_path = os.path.join(ckpt_dir, "last.pt")
    start_epoch = 0
    history = []
    # Lower is better only for unit error rate; the accuracy-like metrics
    # improve upward.
    lower_is_better = tcfg.select_metric in ("unit_error_rate", "dev_uer")
    best = float("inf") if lower_is_better else float("-inf")
    step = 0
    stale = 0  # epochs since the last real improvement

    if os.path.exists(resume_path):
        try:
            state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            start_epoch = state["epoch"] + 1
            step = state["step"]
            best = state["best"]
            stale = state.get("stale", 0)
            history = state.get("history", [])
            print(
                f"[resume] picking up from epoch {start_epoch} (step {step}, "
                f"best {tcfg.select_metric}={best:.4f})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - a corrupt resume must not block
            print(f"[resume] ignoring unusable {resume_path}: {exc}", flush=True)
            start_epoch, step, history = 0, 0, []
            stale = 0
            best = float("inf") if lower_is_better else float("-inf")

    dev = []
    if tcfg.dev_utts:
        from .devset import dev_uer, load_dev_set

        t_dev = time.time()
        dev = load_dev_set(lang.code, units, tcfg.dev_utts)
        print(f"[dev] {len(dev)} held-out utterances with human transcripts "
              f"in {time.time() - t_dev:.0f}s", flush=True)
    elif tcfg.select_metric == "dev_uer":
        raise ValueError("select_metric='dev_uer' needs dev_utts > 0")

    for epoch in range(start_epoch, tcfg.epochs):
        model.train()
        t_epoch = time.time()
        run_loss = 0.0
        run_correct = 0
        run_n = 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
                loss = lossf(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1

            mask = y != C.IGNORE_INDEX
            n = int(mask.sum())
            run_loss += float(loss) * n
            run_correct += int((logits.argmax(1)[mask] == y[mask]).sum())
            run_n += n
            if step % tcfg.log_every == 0:
                print(
                    f"[train] ep{epoch} step {step}/{total_steps} "
                    f"loss {run_loss / max(run_n, 1):.4f} "
                    f"frame_acc {run_correct / max(run_n, 1):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        val = evaluate(model, loaders["val"], device, n_classes)
        if dev:
            val.update(
                dev_uer(model, dev, mean, std, device, sil_class=units.index(C.SIL_TOKEN))
            )
        row = {
            "epoch": epoch,
            "train_loss": run_loss / max(run_n, 1),
            "train_frame_acc": run_correct / max(run_n, 1),
            "val_loss": val["loss"],
            "val_frame_acc": val["frame_acc"],
            "val_balanced_acc": val["balanced_acc"],
            "val_macro_f1": val["macro_f1"],
            "val_unit_error_rate": val["unit_error_rate"],
            "sec": round(time.time() - t_epoch, 1),
        }
        if dev:
            row["dev_uer"] = val["dev_uer"]
            row["dev_len_ratio"] = val["dev_len_ratio"]
        history.append(row)
        print(f"[eval] {json.dumps(row)}", flush=True)

        if tcfg.select_metric not in val:
            raise ValueError(
                f"select_metric={tcfg.select_metric!r} is not one of {sorted(k for k, v in val.items() if isinstance(v, float))}"
            )
        current = val[tcfg.select_metric]
        # `min_delta` guards against counting noise as progress.
        if lower_is_better:
            improved = current < best - tcfg.min_delta
        else:
            improved = current > best + tcfg.min_delta
        stale = 0 if improved else stale + 1
        if improved:
            best = current

        # Written every epoch, so a preemption loses at most one epoch -- and
        # written *after* selection, because `best` and `stale` are what the
        # resumed run selects and early-stops against. Saving before selection
        # would resume a run comparing epoch N+1 against epoch N-1's best,
        # letting a worse epoch overwrite `best.pt`.
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "step": step,
                "best": best,
                "stale": stale,
                "history": history,
            },
            resume_path,
        )

        # One payload, written to possibly two places: `best.pt` is whatever
        # the selection metric liked, and `epoch_NN.pt` is an unconditional
        # per-epoch archive. The archive exists because `best.pt` is destroyed
        # by the next improvement -- on the 30-epoch run the lowest-UER epoch
        # was overwritten four epochs later and could never be scored on the
        # held-out set. At ~46 MB an epoch, keeping them all is far cheaper
        # than being unable to reconsider the choice afterwards.
        payload = {
            "model": model.state_dict(),
            "vocab": units,
            "n_classes": n_classes,
            "language": lang.code,
            "provenance": prov,
            "norm": {"mean": mean, "std": std},
            "unit_median_frames": median_frames,
            "chunk_config": ccfg.to_dict(),
            "train_config": tcfg.to_dict(),
            "receptive_field_ms": model.receptive_field_ms(),
            "feature_config": {
                "sample_rate": C.SAMPLE_RATE,
                "n_fft": C.N_FFT,
                "hop_length": C.HOP_LENGTH,
                "n_mels": C.N_MELS,
                "f_min": C.F_MIN,
                "f_max": C.F_MAX,
            },
            "epoch": epoch,
            "select_metric": tcfg.select_metric,
            "val": {k: v for k, v in val.items() if not isinstance(v, np.ndarray)},
        }

        if tcfg.keep_epoch_checkpoints:
            archive = os.path.join(ckpt_dir, "epochs")
            os.makedirs(archive, exist_ok=True)
            torch.save(payload, os.path.join(archive, f"epoch_{epoch:02d}.pt"))

        if improved:
            torch.save(payload, os.path.join(ckpt_dir, "best.pt"))
            print(f"[ckpt] saved best ({tcfg.select_metric}={best:.4f})", flush=True)
        else:
            print(
                f"[early] no {tcfg.select_metric} gain > {tcfg.min_delta} for "
                f"{stale}/{tcfg.patience} epochs (best {best:.4f})",
                flush=True,
            )

        if tcfg.patience and stale >= tcfg.patience:
            print(
                f"[early] stopping at epoch {epoch}: {stale} epochs without a "
                f"{tcfg.min_delta} gain in {tcfg.select_metric}",
                flush=True,
            )
            break

    # Test the checkpoint that will actually be released, not whatever the last
    # epoch happened to leave in memory. With selection on a metric that can
    # peak before the final epoch, the two differ -- and a model card carrying
    # the final epoch's test numbers beside the selected epoch's weights is
    # simply wrong.
    tested_epoch = history[-1]["epoch"] if history else None
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        chosen = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(chosen["model"])
        tested_epoch = chosen.get("epoch", tested_epoch)
        print(
            f"[test] scoring the selected checkpoint (epoch {tested_epoch}, "
            f"{tcfg.select_metric}={best:.4f})",
            flush=True,
        )
    test = evaluate(model, loaders["test"], device, n_classes)
    summary = {
        "run_name": tcfg.run_name,
        "epochs_run": len(history),
        "tested_epoch": tested_epoch,
        "early_stopped": bool(tcfg.patience and stale >= tcfg.patience),
        "n_classes": n_classes,
        "units": units,
        "receptive_field_ms": model.receptive_field_ms(),
        "n_params": model.n_params(),
        "chunks": {k: int(len(v)) for k, v in subsets.items()},
        "index_stats": index["stats"],
        "history": history,
        "test": {k: v for k, v in test.items() if not isinstance(v, np.ndarray)},
        "test_per_class": {
            units[i]: {
                "recall": round(float(test["recall"][i]), 4),
                "support_frames": int(test["support"][i]),
            }
            for i in range(n_classes)
        },
        "test_contrasts": contrast_report(test, units, lang),
        "provenance": prov,
    }
    with open(os.path.join(ckpt_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    np.save(os.path.join(ckpt_dir, "test_confusion.npy"), test["confusion"])

    print("[test] " + json.dumps(summary["test"]), flush=True)
    print(
        "[test] contrasts " + json.dumps(summary["test_contrasts"], ensure_ascii=False, indent=2),
        flush=True,
    )
    return summary
