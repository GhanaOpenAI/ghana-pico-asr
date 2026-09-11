"""Turn a resume checkpoint into a releasable one.

`last.pt` holds weights plus optimiser and scheduler state, but none of the
metadata inference needs -- vocabulary, normalisation, feature config. `best.pt`
holds all of that, for whichever epoch selection preferred. When the two
disagree, the final epoch's weights are unusable on their own even if they
score better, because nothing records what its 37 outputs mean.

This grafts one onto the other: `best.pt`'s metadata, `last.pt`'s weights. It
is not a substitute for selection -- use it to *measure* a rejected epoch on a
held-out set, and release it only if it actually wins there.
"""

from __future__ import annotations

WEIGHT_KEY = "model"
#: Resume-only state, meaningless in a released checkpoint.
DROP = ("opt", "sched", "step", "stale", "best")


def releasable_from_resume(best: dict, last: dict) -> dict:
    """Return `best`'s metadata carrying `last`'s weights.

    Raises if the two describe different architectures, which would otherwise
    surface as an opaque shape error at load time.
    """
    for name, ck in (("best", best), ("last", last)):
        if WEIGHT_KEY not in ck:
            raise ValueError(f"{name} checkpoint has no {WEIGHT_KEY!r}")

    bw, lw = best[WEIGHT_KEY], last[WEIGHT_KEY]
    if set(bw) != set(lw):
        only_b = sorted(set(bw) - set(lw))[:3]
        only_l = sorted(set(lw) - set(bw))[:3]
        raise ValueError(
            f"architecture mismatch: {len(set(bw) ^ set(lw))} differing tensors "
            f"(best-only {only_b}, last-only {only_l})"
        )
    bad = [k for k in bw if tuple(bw[k].shape) != tuple(lw[k].shape)]
    if bad:
        raise ValueError(f"shape mismatch on {len(bad)} tensors, e.g. {bad[:3]}")

    out = {k: v for k, v in best.items() if k not in DROP}
    out[WEIGHT_KEY] = lw
    out["epoch"] = last.get("epoch", best.get("epoch"))
    history = last.get("history") or []
    row = next((r for r in history if r.get("epoch") == out["epoch"]), None)
    if row is not None:
        # Same shape as what the trainer stores, minus the `val_` prefixes its
        # history rows carry.
        out["val"] = {
            k[len("val_"):]: v for k, v in row.items() if k.startswith("val_")
        }
    out["promoted_from"] = {
        "reason": "final-epoch weights, metadata from the selected checkpoint",
        "selected_epoch": best.get("epoch"),
        "promoted_epoch": out["epoch"],
    }
    return out
