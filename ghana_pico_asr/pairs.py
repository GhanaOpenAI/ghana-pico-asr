"""Training pairs for the text-recovery (T5) stage.

The recogniser emits grapheme units; a second model turns those into words.
That second model needs ``(units, text)`` pairs, and *which* units matters:

**Real pairs** come from running the recogniser on audio whose transcript we
know. Their unit side carries the model's actual error profile — measured at
71% match, 15% substituted, 14% deleted, 4% inserted — so a model trained on
them learns to insert missing units and to doubt vowel quality, which is the
job.

**Synthetic pairs** come from raw text via the deterministic segmenter. Their
unit side is clean, so they teach vocabulary and phrasing for a new domain
without needing any audio — but on their own they would teach the recovery
model that its input is trustworthy, undoing the repair behaviour.

Mixing the two is what makes cheap domain adaptation work: synthetic pairs
supply the new domain, real pairs hold the repair behaviour in place.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass

@dataclass
class Pair:
    """One training example for the text-recovery model."""

    units: str  # model input: space-separated grapheme units
    text: str  # target: the sentence
    origin: str  # "real" (from audio) or "synthetic" (from text)
    source: str = ""  # dataset id / filename it came from
    id: str = ""
    #: Only on real pairs: the units the *reference text* would produce, so the
    #: gap between this and `units` is inspectable.
    reference_units: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["reference_units"] is None:
            d.pop("reference_units")
        return d


def read_pairs(path: str) -> list[Pair]:
    """Load pairs from ``.csv`` or ``.jsonl``, tolerating extra columns."""
    if path.lower().endswith(".csv"):
        return _read_csv(path)
    out: list[Pair] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: not valid JSON ({exc})") from None
            if "units" not in d or "text" not in d:
                raise ValueError(
                    f"{path}:{line_no}: a pairs file needs 'units' and 'text' keys, got {sorted(d)}"
                )
            out.append(
                Pair(
                    units=d["units"],
                    text=d["text"],
                    origin=d.get("origin", "real"),
                    source=d.get("source", ""),
                    id=str(d.get("id", "")),
                    reference_units=d.get("reference_units"),
                )
            )
    return out


def _read_csv(path: str) -> list[Pair]:
    out: list[Pair] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"units", "text"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: a pairs CSV needs {sorted(missing)} column(s); "
                f"found {reader.fieldnames}"
            )
        for row_no, row in enumerate(reader, 2):
            if not (row.get("units") and row.get("text")):
                continue
            out.append(
                Pair(
                    units=row["units"],
                    text=row["text"],
                    origin=row.get("origin") or "real",
                    source=row.get("source") or "",
                    id=row.get("id") or str(row_no),
                    reference_units=row.get("reference_units") or None,
                )
            )
    return out


def write_csv(pairs: list[Pair], path: str) -> None:
    """Write pairs as CSV — the format the published pair set uses."""
    cols = ["units", "text", "origin", "source", "id", "reference_units"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in pairs:
            row = asdict(p)
            row["reference_units"] = row.get("reference_units") or ""
            w.writerow(row)


#: Where the pair set lives. The Hub is the only source: the set is ~294k rows
#: of model output, which is data rather than code and does not belong in a git
#: repo, and a bundled sample would drift out of step with the released model
#: every time the recogniser is retrained.
HF_PAIRS_REPO = {"twi": "ghanaopenai/twi-grapheme-unit-pairs"}


def read_hf_pairs(repo_id: str, split: str = "train", limit: int = 0) -> list[Pair]:
    """Load pairs from a Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            f"loading pairs from {repo_id} needs:  pip install datasets\n"
            "  (or pass --real PATH for a local file)"
        ) from None

    ds = load_dataset(repo_id, split=split, streaming=True)
    cols = set(ds.features or {})
    for need in ("units", "text"):
        if need not in cols:
            raise SystemExit(
                f"{repo_id} has no {need!r} column; a pairs dataset needs "
                f"'units' and 'text'. Columns: {sorted(cols)}"
            )
    if limit:
        ds = ds.take(limit)
    out = []
    for i, row in enumerate(ds):
        if not (row.get("units") and row.get("text")):
            continue
        out.append(
            Pair(
                units=row["units"],
                text=row["text"],
                origin=row.get("origin") or "real",
                source=row.get("source") or "",
                id=str(row.get("id") or i),
                reference_units=row.get("reference_units") or None,
            )
        )
    return out


def load_real_pairs(
    ref: str | None = None, language: str = "twi", limit: int = 0
) -> tuple[list[Pair], str]:
    """Load real pairs from a file or a Hugging Face dataset.

    ``ref`` may be a local ``.csv``/``.jsonl`` path or an HF dataset id. With
    nothing given, the language's published set is pulled from the Hub.

    Returns the pairs and a description of where they came from.
    """
    if ref:
        if os.path.exists(ref):
            return read_pairs(ref), ref
        if "/" in ref:
            return read_hf_pairs(ref, limit=limit), f"hf:{ref}"
        raise SystemExit(
            f"--real {ref!r} is neither an existing file nor an HF dataset id "
            "(which contains '/')"
        )

    repo = HF_PAIRS_REPO.get(language)
    if repo:
        return read_hf_pairs(repo, limit=limit), f"hf:{repo}"

    raise SystemExit(
        f"no published pairs for {language!r}: nothing in pairs.HF_PAIRS_REPO.\n"
        "  Pass --real PATH for a local file, or --real org/dataset."
    )


def write_pairs(pairs: list[Pair], path: str | None) -> None:
    import sys

    fh = open(path, "w", encoding="utf-8") if path else sys.stdout
    try:
        for p in pairs:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    finally:
        if path:
            fh.close()


def _stratified_sample(
    real: list[Pair], take: int, balance: str, rng: random.Random
) -> tuple[list[Pair], dict]:
    """Draw ``take`` pairs spread across the corpora they came from.

    The corpus is heavily unbalanced — film dialogue is ~75% of it — so a
    uniform draw would make the replay three-quarters one domain. ``balance``
    picks the intent:

    ``proportional``  mirror each source's share of the full set. Represents
                      the real error profile as it actually occurs.
    ``equal``         same count per source, so small but clean corpora are not
                      drowned out.
    """
    from collections import Counter, defaultdict

    by_source: dict[str, list[Pair]] = defaultdict(list)
    for p in real:
        by_source[p.source or "?"].append(p)

    sources = sorted(by_source)
    if balance == "equal":
        quota = {s: take // len(sources) for s in sources}
        for s in sources[: take % len(sources)]:
            quota[s] += 1
    elif balance == "proportional":
        quota = {s: int(round(take * len(by_source[s]) / len(real))) for s in sources}
    else:
        raise ValueError(f"balance must be 'proportional' or 'equal', got {balance!r}")

    picked: list[Pair] = []
    shortfall = 0
    for s in sources:
        pool = by_source[s]
        want = quota[s]
        if want >= len(pool):
            shortfall += want - len(pool)
            picked += pool
        else:
            picked += rng.sample(pool, want)

    # Redistribute what small sources could not supply, so the total is honoured.
    if shortfall:
        # Identity, not equality: two pairs can have identical fields, and an
        # `in list` membership test here would be O(n^2) on 200k pairs.
        chosen = {id(p) for p in picked}
        remaining = [p for s in sources for p in by_source[s] if id(p) not in chosen]
        picked += rng.sample(remaining, min(shortfall, len(remaining)))

    return picked, dict(Counter(p.source or "?" for p in picked))


def mix(
    synthetic: list[Pair],
    real: list[Pair],
    ratio: float = 1.0,
    seed: int = 0,
    shuffle: bool = True,
    balance: str = "proportional",
) -> tuple[list[Pair], dict]:
    """Combine new-domain synthetic pairs with a replay sample of real pairs.

    ``ratio`` is parts real per part synthetic. At the default 1.0 the mix is
    even. When there are fewer real pairs available than the ratio asks for,
    **all** of them are used rather than sampling with replacement — repeating
    the same real examples would over-weight them without adding information.

    ``balance`` spreads the replay across source corpora; see
    :func:`_stratified_sample`.

    Returns the mixed list and a report of what the mix actually is, since the
    achieved ratio can differ from the requested one.
    """
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0, got {ratio}")
    want = int(round(len(synthetic) * ratio))
    capped = want > len(real)
    take = min(want, len(real))

    rng = random.Random(seed)
    if take == len(real):
        replay, replay_sources = real, None
    else:
        replay, replay_sources = _stratified_sample(real, take, balance, rng)

    mixed = list(synthetic) + list(replay)
    if shuffle:
        rng.shuffle(mixed)

    from collections import Counter

    report = {
        "synthetic": len(synthetic),
        "real_requested": want,
        "real_available": len(real),
        "real_used": len(replay),
        "real_capped_by_availability": capped,
        "requested_ratio": ratio,
        "achieved_ratio": round(len(replay) / max(len(synthetic), 1), 4),
        "total": len(mixed),
        "shuffled": shuffle,
        "seed": seed,
        "replay_balance": balance if replay_sources is not None else "all-used",
        "replay_by_source": replay_sources
        or dict(Counter(p.source or "?" for p in replay)),
        "available_by_source": dict(Counter(p.source or "?" for p in real)),
    }
    return mixed, report


def describe_mix(report: dict) -> str:
    lines = [
        f"synthetic (new domain)  {report['synthetic']:>8,}",
        f"real (replay)           {report['real_used']:>8,}"
        f"  of {report['real_available']:,} available",
        f"total                   {report['total']:>8,}",
        f"ratio real:synthetic    {report['achieved_ratio']:.3f}"
        f" (requested {report['requested_ratio']})",
    ]
    if report.get("replay_by_source"):
        lines.append(
            "replay by source        "
            + "  ".join(f"{k}={v:,}" for k, v in sorted(report["replay_by_source"].items()))
            + f"   ({report.get('replay_balance')})"
        )
    if report["real_capped_by_availability"]:
        lines.append(
            f"note: wanted {report['real_requested']:,} real pairs but only "
            f"{report['real_available']:,} exist, so all were used. Generate more "
            "real pairs with `pico make-pairs` to reach the requested ratio."
        )
    return "\n".join(lines)
