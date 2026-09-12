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

#: Clean Twi text for building error-free pairs. Its register — news and
#: conversation — is deliberately unlike the training corpora (scripture, health
#: talk shows, film dialogue), because the failure this addresses is domain, not
#: noise: on held-out Waxal the recovery models reconstructed fluent sentences
#: from the *training* distribution rather than the one they were given.
CLEAN_TEXT_REPO = "ghananlpcommunity/pristine-twi-english-parallel-sentences"
CLEAN_TEXT_COLUMN = "twi"


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

    #: Extra pairs built from the training samples' own `reference_units`,
    #: as a fraction of the real pairs. 0 disables.
    #:
    #: Same sentence, correct units: the model sees the utterance it will also
    #: meet corrupted, paired with what the units *should* have been. That
    #: isolates the restoration half on exactly the material the recogniser
    #: gets wrong.
    #:
    #: The reference units are the target text with word boundaries,
    #: capitalisation, apostrophes and punctuation stripped, so a clean pair
    #: teaches *only* the restoration half of the job, with no errors to
    #: correct. Real pairs teach restoration and correction at once, which is
    #: harder to learn from alone.
    #:
    clean_ratio: float = 0.0

    #: Extra pairs built from an outside text corpus, as a fraction of the real
    #: pairs. Independent of `clean_ratio`: the two teach different things and
    #: are meant to be combined.
    #:
    #: Same-corpus clean pairs fix the mapping but leave the text distribution
    #: untouched — and that distribution is what failed to transfer to Waxal.
    #: Outside text is the only source of new *sentences*.
    #:
    #: Both are clean, so neither teaches correction. Weighting them heavily
    #: risks a model that formats well and corrects poorly, since at inference
    #: it only ever sees noisy units.
    clean_text_ratio: float = 0.0

    #: Which corpus outside text comes from.
    clean_text_repo: str = CLEAN_TEXT_REPO


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


def clean_pairs_from_text(
    n: int,
    language: str = "twi",
    repo_id: str | None = None,
    column: str | None = None,
    spaced: bool = False,
    min_units: int = 8,
    seed: int = 0,
) -> list[dict]:
    """Build (units, text) pairs from raw text, with no recogniser involved.

    `lang.segment` is deterministic, so any Twi text becomes a training pair
    for free — no audio, no GPU. These carry no errors to correct, so they
    teach restoration only: word boundaries, capitalisation, apostrophes and
    punctuation. That is the domain-independent half of the job, and the half
    that transferred worst.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ..languages import flatten, get_language

    lang = get_language(language)
    path = hf_hub_download(
        repo_id or CLEAN_TEXT_REPO,
        "data/train-00000-of-00006.parquet",
        repo_type="dataset",
    )
    col = column or CLEAN_TEXT_COLUMN
    out: list[dict] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=[col]):
        if len(out) >= n:
            break
        for text in batch.column(col).to_pylist():
            if len(out) >= n:
                break
            text = (text or "").strip()
            if not text:
                continue
            units = flatten(lang.segment(text))
            if len(units) < min_units:
                continue
            out.append(
                {
                    "units": " ".join(units),
                    "reference_units": " ".join(units),
                    "text": text,
                    "source_text": format_source(" ".join(units), spaced=spaced),
                    "target_text": text,
                    "origin": "clean-text",
                    "source": -1,
                    "uer": 0.0,
                    "id": f"clean{len(out):07d}",
                }
            )
    return out


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
        # Clean pairs never leave the training set. Their input is already
        # correct, so scoring on them measures formatting on free wins and
        # deflates both CER and the baseline — a 2:1 clean mix moved the
        # reported baseline from 0.4395 to 0.2876 while nothing about the task
        # had changed.
        if r.get("origin") not in (None, "real"):
            out["train"].append(r)
            continue
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
        r["origin"] = "real"

    extra: list[dict] = []

    if flt.clean_ratio > 0:
        # The training sentences again, with the units they should have had.
        pool = [r for r in kept if (r.get("reference_units") or "").strip()]
        rng = random.Random(seed + 1)
        for r in rng.sample(pool, min(int(len(kept) * flt.clean_ratio), len(pool))):
            c = dict(r)
            c["source_text"] = format_source(r["reference_units"], spaced=spaced)
            c["origin"] = "clean-ref"
            c["uer"] = 0.0
            extra.append(c)
        report["clean_ref_added"] = len(extra)

    if flt.clean_text_ratio > 0:
        # New sentences the model has never been asked to produce.
        from_text = clean_pairs_from_text(
            int(len(kept) * flt.clean_text_ratio),
            language=language, repo_id=flt.clean_text_repo, spaced=spaced,
            min_units=flt.min_units, seed=seed,
        )
        extra += from_text
        report["clean_text_added"] = len(from_text)
        report["clean_text_source"] = flt.clean_text_repo

    if extra:
        kept = kept + extra
        random.Random(seed + 2).shuffle(kept)
        report["mix"] = {
            k: sum(1 for r in kept if r.get("origin") == k)
            for k in ("real", "clean-ref", "clean-text")
        }

    report["final"] = len(kept)
    report["spaced_input"] = spaced
    return kept, report
