"""Turning published (units, text) pairs into fine-tuning data.

Three decisions are encoded here, each measured on the published set rather
than assumed.

**Units are joined, not space-separated.** `ɔyɛneho` costs 104 NLLB tokens on
average against 246 for `ɔ y ɛ n e h o`, because the joined form tokenises into
real Twi subwords instead of one token per character. That is 2.4x the compute
for the same data. The cost is that `ky` can no longer be told from `k`+`y` by
position alone; `spaced` keeps the boundaries if an ablation wants them.

**Pairs the recogniser mangled are dropped.** 8.5% of pairs exceed 0.6 unit
error rate against their own reference, and one corpus averages 0.562 with a
p90 of 0.894. Text is not recoverable from input that corrupted, so those pairs
teach the model to invent plausible Twi rather than to correct — the exact
failure mode a recovery stage must not have.

**Machine-transcribed targets are down-weighted.** 63% of pairs carry targets
from Google STT or Gemini, and a recovery model learns to write whatever its
targets say. Training predominantly on those teaches it to reproduce another
recogniser's mistakes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .. import config as C

#: Source ids whose reference text was written by a person. From
#: `config.SOURCE_IDS`; ids 2 and 3 are machine transcripts.
HUMAN_SOURCES = frozenset({1, 4, 5, 7})

#: NLLB language codes. Twi is a dialect of Akan and both are in NLLB-200, so
#: which one conditions better is an empirical question, not a settled one.
NLLB_LANG = "twi_Latn"
NLLB_LANG_ALT = "aka_Latn"


@dataclass
class PairFilter:
    """Which pairs are worth training on."""

    #: Drop pairs whose units differ from the reference by more than this.
    #: 0.5 keeps ~86% of the set; 1.0 disables the filter.
    max_uer: float = 0.5
    #: Drop pairs shorter than this many units — too little signal to learn a
    #: mapping from, and they dominate a per-utterance average.
    min_units: int = 8
    #: Keep at most this multiple of the human-transcribed count from
    #: machine-transcribed sources. 0 drops them entirely, 0.5 lets them
    #: augment without dominating, and a large value keeps everything.
    machine_ratio: float = 0.5
    #: Sources to exclude outright, by id.
    exclude_sources: frozenset = field(default_factory=frozenset)


def pair_uer(units: str, reference_units: str) -> float:
    """Unit error rate of one pair against its own reference."""
    from ..devset import edit_distance

    ref = reference_units.split()
    if not ref:
        return 1.0
    return edit_distance(units.split(), ref) / len(ref)


def format_source(units: str, spaced: bool = False) -> str:
    """The encoder's input string for a unit sequence."""
    return units if spaced else units.replace(" ", "")


def _bucket(key: str) -> int:
    """Stable 0-99 bucket, so a split survives regeneration and reordering."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def split_pairs(rows: list[dict], val_pct: int = 2, test_pct: int = 2) -> dict:
    """Hash-bucket split keyed on the reference text.

    Keyed on the *text* rather than the row id: the same sentence can appear in
    more than one corpus, and a recovery model that saw it in training would be
    scored on a sentence it has memorised.
    """
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        b = _bucket(r["text"])
        if b < val_pct:
            out["val"].append(r)
        elif b < val_pct + test_pct:
            out["test"].append(r)
        else:
            out["train"].append(r)
    return out


def _uer_column(rows: list[dict], cache_dir: str | None, key: str) -> list[float]:
    """Per-pair UER for every row, cached.

    Scoring 282k pairs by edit distance takes ~12 minutes, and it is the same
    answer every run — including for a 4k smoke run that then throws almost all
    of it away. Cached beside the checkpoints so only the first run pays.
    """
    import json
    import os

    path = os.path.join(cache_dir, f"uer_{key}.json") if cache_dir else None
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                cached = json.load(fh)
            if len(cached) == len(rows):
                return cached
        except Exception:  # noqa: BLE001 - a bad cache must not block a run
            pass

    out = [
        pair_uer(r["units"], r["reference_units"]) if r.get("reference_units") else 0.0
        for r in rows
    ]
    if path:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh)
        except OSError:
            pass
    return out


def load_pairs(
    repo_id: str | None = None,
    language: str = "twi",
    flt: PairFilter | None = None,
    limit: int = 0,
    spaced: bool = False,
    seed: int = 0,
    cache_dir: str | None = None,
) -> tuple[list[dict], dict]:
    """Load, filter and balance the published pairs.

    Returns the rows (each with `source_text` and `target_text`) and a report
    of what was dropped and why, so a training run records the data it saw.
    """
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ..pairs import HF_PAIRS_REPO

    flt = flt or PairFilter()
    repo_id = repo_id or HF_PAIRS_REPO.get(language)
    if not repo_id:
        raise SystemExit(f"no published pairs for {language!r}")

    path = hf_hub_download(
        repo_id, "data/train-00000-of-00001.parquet", repo_type="dataset"
    )
    table = pq.read_table(path)
    rows = table.to_pylist()
    report = {"repo": repo_id, "loaded": len(rows)}

    uers = (
        _uer_column(rows, cache_dir, f"{repo_id.replace('/', '_')}_{len(rows)}")
        if flt.max_uer < 1.0
        else [0.0] * len(rows)
    )

    kept, drop_short, drop_uer, drop_src = [], 0, 0, 0
    for r, u in zip(rows, uers):
        if r.get("source") in flt.exclude_sources:
            drop_src += 1
            continue
        n_units = len(r["units"].split())
        if n_units < flt.min_units or not r["text"].strip():
            drop_short += 1
            continue
        if flt.max_uer < 1.0 and r.get("reference_units"):
            if u > flt.max_uer:
                drop_uer += 1
                continue
            r["uer"] = round(u, 4)
        kept.append(r)

    report.update(
        {"dropped_excluded_source": drop_src, "dropped_short": drop_short,
         "dropped_high_uer": drop_uer, "after_filter": len(kept)}
    )

    human = [r for r in kept if r.get("source") in HUMAN_SOURCES]
    machine = [r for r in kept if r.get("source") not in HUMAN_SOURCES]
    cap = int(len(human) * flt.machine_ratio)
    if len(machine) > cap:
        random.Random(seed).shuffle(machine)
        machine = machine[:cap]
    kept = human + machine
    report.update({"human": len(human), "machine_kept": len(machine)})

    random.Random(seed).shuffle(kept)
    if limit:
        kept = kept[:limit]

    for r in kept:
        r["source_text"] = format_source(r["units"], spaced=spaced)
        r["target_text"] = r["text"].strip()
    report["final"] = len(kept)
    report["spaced_input"] = spaced
    return kept, report
