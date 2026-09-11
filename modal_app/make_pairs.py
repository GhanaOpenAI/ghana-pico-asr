"""Transcribe the whole training corpus to build real (units, text) pairs.

These are the training set for the text-recovery model: the recogniser's own
output paired with the reference transcript, so the recovery model learns to
repair the errors it will actually meet.

This runs on the **stored mel spectrograms**, not the source audio — stage 1
already wrote them to the volume, and the reference text is in the per-shard
manifests. So there is no audio decoding, no resampling and no Hugging Face
download: just conv forward passes over memmapped arrays. 584 h of audio takes
~20 min here versus ~15 h of local CPU.

    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/make_pairs.py --run-name final
    modal run modal_app/make_pairs.py --run-name final --sample-size 25000
    modal run modal_app/make_pairs.py --run-name final --push-hf
    modal volume get twi-phoneme pairs/real_pairs_twi.csv .

The full set — every utterance, no subsampling — is published to Hugging Face,
since it is a dataset rather than code and is too large for a git repo. The Hub
copy is the only one: `pico finetune-data` pulls from it, rather than from a
bundled sample that would drift out of step with the released recogniser.
"""

import json

import modal

from modal_app.common import VOLUMES, app, hf_secret, image, volume
from ghana_pico_asr import config as C

image = image.add_local_python_source("modal_app")

PAIRS_DIR = "pairs"


@app.function(
    image=image,
    gpu="a10g",
    volumes=VOLUMES,
    timeout=60 * 60 * 2,
    memory=16384,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
    max_containers=10,
)
def pairs_for_shard(job: dict) -> dict:
    """One feature-store shard -> one CSV of pairs on the volume."""
    import os
    import time

    import numpy as np
    import torch

    from ghana_pico_asr import dataset as D
    from ghana_pico_asr.infer import UnitTagger
    from ghana_pico_asr.languages import flatten
    from ghana_pico_asr.pairs import Pair, write_csv
    from ghana_pico_asr.trainer import edit_distance

    stem = job["stem"]
    out_path = os.path.join(C.VOLUME_MOUNT, PAIRS_DIR, f"{stem}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not job.get("overwrite"):
        return {"stem": stem, "status": "cached"}

    ckpt = os.path.join(C.VOLUME_MOUNT, C.CKPT_DIR, job["run_name"], "best.pt")
    tagger = UnitTagger(ckpt, device="cuda")
    lang = tagger.language

    mel_path = os.path.join(C.VOLUME_MOUNT, C.MEL_DIR, f"{stem}.npy")
    npz_path = os.path.join(C.VOLUME_MOUNT, C.MANIFEST_DIR, f"{stem}.npz")
    jsonl_path = os.path.join(C.VOLUME_MOUNT, C.MANIFEST_DIR, f"{stem}.jsonl")

    mel = np.load(mel_path, mmap_mode="r")
    with np.load(npz_path) as z:
        offsets = z["utt_offsets"].astype(np.int64)
        nframes = z["utt_nframes"].astype(np.int64)
        utt_scores = z["utt_scores"].astype(np.float32)

    # The manifest jsonl carries the reference transcript per utterance.
    records = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) != len(offsets):
        return {
            "stem": stem,
            "status": "error",
            "error": f"manifest has {len(records)} records but {len(offsets)} utterances",
        }

    t0 = time.time()
    mean, std = tagger.mean, tagger.std
    batch_frames = job.get("batch_frames", 60_000)

    # Sort by length so padding within a batch is small; the model is
    # convolutional, so a padded tail is equivalent to the zero padding it
    # would see if the utterance were run alone.
    order = np.argsort(nframes)
    pairs: list[Pair] = []
    skipped = {"short": 0, "no_text": 0, "low_score": 0, "high_uer": 0}
    tot_err = tot_ref = 0

    i = 0
    while i < len(order):
        batch_idx, longest = [], 0
        while i < len(order):
            k = int(order[i])
            n = int(nframes[k])
            cand = max(longest, n)
            if batch_idx and cand * (len(batch_idx) + 1) > batch_frames:
                break
            batch_idx.append(k)
            longest = cand
            i += 1

        keep = []
        for k in batch_idx:
            rec = records[k]
            text = (rec.get("text") or "").strip()
            if not text:
                skipped["no_text"] += 1
                continue
            if utt_scores[k] < job.get("min_utt_score", -2.0):
                skipped["low_score"] += 1
                continue
            ref = flatten(lang.segment(text))
            if len(ref) < job.get("min_units", 5):
                skipped["short"] += 1
                continue
            keep.append((k, text, ref))
        if not keep:
            continue

        x = torch.zeros(len(keep), 1, C.N_MELS, longest, dtype=torch.float32)
        for b, (k, _, _) in enumerate(keep):
            off, n = int(offsets[k]), int(nframes[k])
            seg = np.asarray(mel[off : off + n], dtype=np.float32)
            x[b, 0, :, :n] = torch.from_numpy((seg - mean) / std).T

        with torch.inference_mode():
            logits = tagger.model(x.to("cuda"))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

        for b, (k, text, ref) in enumerate(keep):
            n = int(nframes[k])
            units = [
                u.unit
                for u in tagger.units_from_posteriors(
                    probs[b, :, :n].T,
                    smooth_frames=job.get("smooth_frames", 7),
                    min_frames=job.get("min_frames", 4),
                )
            ]
            err = edit_distance(units, ref)
            uer = err / len(ref)
            if uer > job.get("max_uer", 1.0):
                skipped["high_uer"] += 1
                continue
            tot_err += err
            tot_ref += len(ref)
            pairs.append(
                Pair(
                    units=" ".join(units),
                    text=text,
                    origin="real",
                    # Numeric, not the corpus name — see config.SOURCE_IDS.
                    source=str(C.source_id(stem.split("_")[0])),
                    id=records[k].get("utt", f"{stem}_{k}"),
                    reference_units=" ".join(ref),
                )
            )

    write_csv(pairs, out_path)
    volume.commit()
    return {
        "stem": stem,
        "status": "ok",
        "pairs": len(pairs),
        "uer": round(tot_err / max(tot_ref, 1), 4),
        "ref_units": tot_ref,
        "skipped": {k: v for k, v in skipped.items() if v},
        "sec": round(time.time() - t0, 1),
    }


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60, cpu=4.0, memory=32768)
def finalise(language: str, sample_size: int, seed: int) -> dict:
    """Concatenate the per-shard CSVs and write a stratified sample.

    The full set is for training the text-recovery model; the sample is small
    enough to commit to the repo, for replay mixing with no external download.
    """
    import csv
    import os
    import random
    from collections import Counter, defaultdict

    src_dir = os.path.join(C.VOLUME_MOUNT, PAIRS_DIR)
    shard_files = sorted(
        f for f in os.listdir(src_dir) if f.endswith(".csv") and not f.startswith("real_pairs")
    )
    if not shard_files:
        return {"error": "no per-shard CSVs found; run pairs_for_shard first"}

    cols = ["units", "text", "origin", "source", "id", "reference_units"]
    full = os.path.join(src_dir, f"real_pairs_{language}.csv")
    by_source: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()

    with open(full, "w", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for f in shard_files:
            with open(os.path.join(src_dir, f), encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    w.writerow(row)
                    counts[row.get("source", "?")] += 1
                    # Reservoir-free: keep a bounded pool per source for sampling.
                    pool = by_source[row.get("source", "?")]
                    if len(pool) < max(sample_size, 1) * 4:
                        pool.append(row)

    # Stratified sample: proportional to each source's share of the full set.
    rng = random.Random(seed)
    total = sum(counts.values())
    sample_path = os.path.join(src_dir, f"real_pairs_{language}_sample.csv")
    picked: list[dict] = []
    for source, n in counts.items():
        want = int(round(sample_size * n / max(total, 1)))
        pool = by_source[source]
        picked += pool if want >= len(pool) else rng.sample(pool, want)
    rng.shuffle(picked)
    with open(sample_path, "w", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(picked)

    volume.commit()
    return {
        "full_csv": full,
        "full_pairs": total,
        "full_mb": round(os.path.getsize(full) / 1e6, 1),
        "by_source": dict(counts),
        "sample_csv": sample_path,
        "sample_pairs": len(picked),
        "sample_mb": round(os.path.getsize(sample_path) / 1e6, 2),
    }


@app.function(
    image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 90, cpu=4.0, memory=32768
)
def push_hf(language: str, repo_id: str, private: bool) -> dict:
    """Publish the full pair set to Hugging Face as parquet.

    Parquet rather than CSV: typed, ~4x smaller, and the dataset viewer works.
    """
    import os

    import pyarrow as pa
    import pyarrow.csv as pv
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return {"error": "no HF token in the `huggingface` secret"}

    src_dir = os.path.join(C.VOLUME_MOUNT, PAIRS_DIR)
    csv_path = os.path.join(src_dir, f"real_pairs_{language}.csv")
    if not os.path.exists(csv_path):
        return {"error": f"{csv_path} not found; run the transcription step first"}

    table = pv.read_csv(
        csv_path,
        convert_options=pv.ConvertOptions(
            column_types={
                **{c: pa.string() for c in
                   ("units", "text", "origin", "id", "reference_units")},
                "source": pa.int32(),  # a real integer column in the parquet
            }
        ),
    )
    pq_path = os.path.join(src_dir, f"real_pairs_{language}.parquet")
    pq.write_table(table, pq_path, compression="zstd")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    card = f"""---
license: cc-by-nc-4.0
language: [tw, ak]
task_categories: [translation]
tags: [speech, grapheme-units, ghana-pico-asr, text-recovery, twi, akan]
size_categories: [100K<n<1M]
---

# Grapheme-unit / text pairs ({language})

Output of [ghana-pico-asr](https://github.com/ghanaopenai/ghana-pico-asr) run
over its own training corpus, paired with the reference transcript. Training
data for the **text-recovery** stage: a model that turns grapheme-unit
sequences into sentences.

| column | meaning |
|---|---|
| `units` | what the recogniser emitted — **lossy**, this is the model input |
| `text` | the reference transcript — the target |
| `reference_units` | what `text` segments to, for inspecting the gap |
| `source` | numeric id of the corpus it came from (see below) |
| `id` | utterance id in the feature store |

## Source ids

| id | speech | transcripts |
|---|---|---|
| 1 | read speech, studio | human |
| 2 | health talk shows | machine |
| 3 | film dialogue | machine |

Numeric rather than named so the ids stay stable if a corpus is renamed or
replaced. Ids are append-only.

`units` differs from `reference_units` because the recogniser makes mistakes,
and learning to repair exactly those mistakes is the point. Measured error
profile: **71% match, 15% substituted, 14% deleted, 4% inserted**. Deletions
outnumber insertions ~3:1, so a recovery model mostly has to *insert*, and
substitutions are almost entirely vowel-for-vowel (`a→ɛ`, `ɔ→o`, `o→u`).

Every utterance of the recogniser's training corpus is included — no
subsampling.

Non-commercial: the underlying corpora are CC-BY-NC-4.0.
"""
    api.upload_file(
        path_or_fileobj=pq_path,
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    volume.commit()
    return {
        "repo": f"https://huggingface.co/datasets/{repo_id}",
        "rows": table.num_rows,
        "parquet_mb": round(os.path.getsize(pq_path) / 1e6, 1),
        "csv_mb": round(os.path.getsize(csv_path) / 1e6, 1),
        "private": private,
    }


@app.local_entrypoint()
def main(
    run_name: str = "hf30",
    language: str = "twi",
    splits: str = "tts,asr,kuma,female,agric,bible",
    sample_size: int = 25000,
    max_uer: float = 1.0,
    overwrite: bool = False,
    seed: int = 0,
    push_hf_repo: str = "ghanaopenai/twi-grapheme-unit-pairs",
    push: bool = False,
    private: bool = False,
):
    from ghana_pico_asr import dataset as D

    wanted = tuple(s.strip() for s in splits.split(",") if s.strip())
    # Shard listing needs the volume, so ask a remote function for it.
    stems = list_stems.remote(wanted)
    if not stems:
        raise SystemExit(f"no prepared shards for splits {wanted}")
    print(f"transcribing {len(stems)} shards with checkpoint '{run_name}'")

    jobs = [
        {"stem": s, "run_name": run_name, "max_uer": max_uer, "overwrite": overwrite}
        for s in stems
    ]
    results = list(pairs_for_shard.map(jobs, order_outputs=False, return_exceptions=True))

    ok = [r for r in results if r.get("status") == "ok"]
    bad = [r for r in results if r.get("status") == "error"]
    print(f"\n{len(ok)} shards ok, {len(bad)} failed")
    if ok:
        n = sum(r["pairs"] for r in ok)
        refs = sum(r["ref_units"] for r in ok)
        err = sum(r["uer"] * r["ref_units"] for r in ok)
        print(f"  {n:,} pairs | corpus UER {err / max(refs, 1):.4f} over {refs:,} units")
    for r in bad:
        print(f"  FAILED {r['stem']}: {r.get('error')}")

    print("\nconcatenating and sampling...")
    report = finalise.remote(language, sample_size, seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if push:
        print(f"\npublishing the full set to {push_hf_repo} ...")
        print(json.dumps(push_hf.remote(language, push_hf_repo, private),
                         ensure_ascii=False, indent=2))
    else:
        print("\n(pass --push to publish the full set to Hugging Face)")

    print(
        f"\nfetch a local copy if you want one:\n"
        f"  modal volume get {C.VOLUME_NAME} {PAIRS_DIR}/real_pairs_{language}.csv ."
    )


@app.function(image=image, volumes=VOLUMES, timeout=600)
def list_stems(splits: tuple) -> list:
    from ghana_pico_asr import dataset as D

    return [D.shard_stem(npz) for _, npz in D.list_shards(C.VOLUME_MOUNT, splits)]
