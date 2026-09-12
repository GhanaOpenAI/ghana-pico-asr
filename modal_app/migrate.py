"""Move a prepared feature store between Modal workspaces.

Volumes are workspace-local, so a run cannot simply be pointed at another
workspace: the feature store has to travel. Only two things actually need to.

**The store (24.1 GB) goes through a public HF dataset** -- `features/mel` and
`features/manifest`, plus `vocab.json`. Both hops stay inside a datacenter (no
home uplink) on CPU-only containers, so the move costs no GPU credits, and the
same upload is what anyone else needs to train Twi from scratch without
repeating the GPU alignment pass.

**A run's `last.pt` (~190 MB) hops through your machine.** It is the one file
that cannot be regenerated, and it is nobody else's business, so it does not
belong in a public dataset.

`features/index` (676 MB for a six-corpus store) travels *neither* way: it is a
cache the trainer rebuilds from the manifests when absent, and the rebuild is
deterministic -- `list_shards()` sorts, `_cap_per_split()` seeds
`default_rng(0)` -- so the reconstructed chunk index and train/val/test split
are identical, and validation metrics stay comparable across the move. Moving
it would cost bandwidth to arrive at the same bytes.

    # source workspace: publish the store
    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/migrate.py::push --repo-id ghanaopenai/twi-grapheme-unit-features

    # target workspace: pull it onto a fresh volume
    export MODAL_PROFILE=michseth PICO_HF_SECRET=huggingface-secret
    modal run modal_app/migrate.py::pull --repo-id ghanaopenai/twi-grapheme-unit-features

    # the resume checkpoint, via this machine
    MODAL_PROFILE=ghana-nlp modal volume get twi-phoneme checkpoints/xl/last.pt ./hop/
    MODAL_PROFILE=michseth  modal volume put twi-phoneme ./hop/last.pt \
        checkpoints/xl/last.pt

Verify a transfer by size, never by exit status: `modal volume get` on a
directory has been seen to exit 0 having written truncated files.

Then deploy and resume; `last.pt` restarts at the epoch after the one it
recorded, with optimiser, scheduler, best-so-far and history intact:

    PICO_HF_SECRET=huggingface-secret MODAL_PROFILE=michseth \
        modal deploy modal_app/train.py
    MODAL_PROFILE=michseth python scripts/spawn_training.py --run-name xl ...

`which_index` reports which index artifacts a given ChunkConfig resolves to,
if you would rather copy one than rebuild it.
"""

import modal

from modal_app.common import VOLUMES, hf_secret, image, volume
from ghana_pico_asr import config as C

image = image.add_local_python_source("modal_app")

# Its own app, deliberately: every other module shares one app name, so
# deploying this one alongside them would replace the deployment a running
# training call retries into.
MIGRATE_APP = "twi-phoneme-migrate"
app = modal.App(MIGRATE_APP, image=image)

FEATURES = "features"
_HOUR = 60 * 60


def _token() -> str:
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("no HF token in the secret bound to this app")
    return token


CARD = """---
license: cc-by-nc-4.0
language: [tw, ak]
task_categories: [automatic-speech-recognition]
tags: [speech, forced-alignment, phoneme-recognition, mel-spectrogram, twi, akan, ghana-pico-asr]
size_categories: [100K<n<1M]
pretty_name: Twi Grapheme-Unit Feature Store
---

# Twi Grapheme-Unit Feature Store

The training data behind [ghana-pico-asr](https://github.com/ghanaopenai/ghana-pico-asr):
{hours:,.0f} hours of Twi speech turned into log-mel features with a
**grapheme-unit label for every 10 ms frame**, produced by CTC forced
alignment. Publishing it means the expensive step -- aligning {hours:,.0f} hours
on a GPU -- does not have to be repeated to train, reproduce or extend the
model, and the same pipeline can be pointed at a new language.

This is a **derived feature store, not a speech corpus**: it contains 40-band
log-mel spectrograms, not audio. The original recordings live in the source
datasets listed below.

## What a label is

Twi is written with digraphs that behave as single sounds (`ky gy hy ny tw dw
kw gw hw nw`), so the label alphabet is *grapheme units* rather than letters:
each unit is one of those digraphs, a single letter, or `<sil>`, for
{n_classes} classes in total. Text is segmented greedily, longest match first,
so `akyekyedeɛ` becomes `a ky e ky e d e ɛ`.

Alignment used [`{aligner}`]({aligner_url}) at a {stride:g} ms emission stride,
with a `<star>` wildcard allowed at utterance edges and word boundaries only --
never inside a word, where it would happily swallow a real unit.

## Contents

| path | what |
|---|---|
| `features/mel/<source>_<shard>.npy` | float16 log-mel, {n_mels} bands, {frame_ms:g} ms hop, {sr} Hz, one blob per shard |
| `features/manifest/<source>_<shard>.npz` | per-utterance unit ids, frame spans and alignment scores |
| `features/manifest/<source>_<shard>.jsonl` | the same, human-readable, one row per utterance |
| `features/vocab.json` | unit -> class id; **the ordering is part of the model contract** |

The per-frame label tracks and chunk index are *not* shipped: the training code
builds them from the manifests on first run and caches them, and because chunk
assignment comes from hashing utterance ids, the train/validation/test split it
produces is identical to the one the released model was trained on.

## Scale

| | |
|---|---|
| audio | {hours:,.2f} h |
| utterances | {n_utts:,} |
| shards | {n_shards} |
| 1 s chunks (500 ms stride) | {n_chunks:,} |
| labelled frames | {labelled:,} ({pct:.1f}% of frames) |
| classes | {n_classes} |

The remaining frames are deliberately unlabelled (`IGNORE_INDEX`) and
contribute no loss: inter-unit gaps too long to attribute to either neighbour,
and gaps that are loud enough that calling them silence would be a lie.
`<sil>` is {sil_pct:.1f}% of labelled frames.

## Sources

Numeric ids, so the data does not hard-code corpus identities and stays valid
if a corpus is renamed. Ids are append-only.

| id | dataset | transcripts |
|---|---|---|
| 1 | [ghanaopenai/new-twi-tts-aligned](https://huggingface.co/datasets/ghanaopenai/new-twi-tts-aligned) | human |
| 2 | [ghananlpcommunity/twi-health-asr-gemini-500hrs](https://huggingface.co/datasets/ghananlpcommunity/twi-health-asr-gemini-500hrs) | machine (Gemini) |
| 3 | [ghananlpcommunity/kumawood-speech-transcriptions](https://huggingface.co/datasets/ghananlpcommunity/kumawood-speech-transcriptions) | machine (Google STT) |
| 4 | [ghananlpcommunity/ghana-female-twi-asr-16word-splits](https://huggingface.co/datasets/ghananlpcommunity/ghana-female-twi-asr-16word-splits) | human |
| 5 | [ghanaopenai/twi-agriculture-speech](https://huggingface.co/datasets/ghanaopenai/twi-agriculture-speech) | human |
| 7 | [ghanaopenai/asante-twi-bible-speech-text](https://huggingface.co/datasets/ghanaopenai/asante-twi-bible-speech-text) | human |

Id 6 (`ghanaopenai/twi-speech-text-multispeaker-16k`) is reserved but absent:
its audio could not be decoded. A minority of shards from ids 1 and 4 are
likewise missing -- the store holds what aligned cleanly, and shard names are
therefore not contiguous.

Utterances were kept only at {min_dur:g}-{max_dur:g} s, at least {min_units}
units, and {min_ups:g}-{max_ups:g} units/second; a rate outside that band means
the transcript and the audio disagree, which is a real risk with machine
transcripts.

## Using it

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{repo}",
    repo_type="dataset",
    allow_patterns=["features/**"],
    local_dir="/data",          # then train against /data
)
```

Training, evaluation and inference code, including the alignment stage that
produced this store and the language registry for adding another language, is
at **https://github.com/ghanaopenai/ghana-pico-asr**.

## Provenance and licence

**CC-BY-NC-4.0 — non-commercial.** Every source corpus carries that term, and
a derived work cannot be licensed more permissively than its inputs.

The *code* that produced this store is MIT
([ghana-pico-asr](https://github.com/ghanaopenai/ghana-pico-asr)), so running
the same pipeline over a corpus you may use commercially leaves you
unrestricted. Consult the source datasets linked above before redistributing.
No audio is included here.
"""


def _card(repo_id: str, stats: dict) -> str:
    ix = stats.get("index_stats", {})
    labelled = ix.get("labelled_frames", 0)
    return CARD.format(
        repo=repo_id,
        hours=stats.get("audio_hours", 0.0),
        n_utts=stats.get("n_utts", 0),
        n_shards=stats.get("n_shards", 0),
        n_chunks=stats.get("n_chunks", 0),
        labelled=labelled,
        pct=stats.get("frames_labelled_pct", 0.0),
        n_classes=stats.get("n_classes", 0),
        sil_pct=100.0 * ix.get("sil_frames", 0) / max(labelled, 1),
        aligner=C.ALIGNER_MODEL,
        aligner_url=f"https://huggingface.co/{C.ALIGNER_MODEL}",
        stride=C.ALIGNER_STRIDE_MS,
        n_mels=C.N_MELS,
        frame_ms=C.FRAME_MS,
        sr=C.SAMPLE_RATE,
        min_dur=C.MIN_DURATION_S,
        max_dur=C.MAX_DURATION_S,
        min_units=C.MIN_UNITS,
        min_ups=C.MIN_UNITS_PER_SEC,
        max_ups=C.MAX_UNITS_PER_SEC,
    )


def _tree(root: str) -> dict:
    """Name -> size for every file under `root`, for verifying a transfer."""
    import os

    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            out[os.path.relpath(p, root)] = os.path.getsize(p)
    return out


@app.function(image=image, volumes=VOLUMES, timeout=15 * 60, cpu=2.0)
def which_index(ccfg_kwargs: dict) -> dict:
    """Name the index artifacts a given ChunkConfig resolves to.

    The store accumulates one index per (config, store) pair, so picking the
    right one out of nine by eye -- or by modification time -- is guesswork.
    This asks the same function the trainer uses.
    """
    import os

    from ghana_pico_asr import dataset as D

    key = D.cache_key(C.VOLUME_MOUNT, C.ChunkConfig(**ccfg_kwargs))
    ix = os.path.join(C.VOLUME_MOUNT, C.INDEX_DIR)
    wanted = {
        "labels": os.path.join(ix, f"labels_{key}"),
        "chunks": os.path.join(ix, f"chunks_{key}.npz"),
        "vocab": os.path.join(ix, f"vocab_{key}.json"),
        "norm": os.path.join(ix, f"norm_{key}.json"),
    }
    out = {"cache_key": key, "artifacts": {}}
    for name, path in wanted.items():
        if os.path.isdir(path):
            files = _tree(path)
            out["artifacts"][name] = {
                "path": os.path.relpath(path, C.VOLUME_MOUNT),
                "files": len(files),
                "bytes": sum(files.values()),
            }
        elif os.path.exists(path):
            out["artifacts"][name] = {
                "path": os.path.relpath(path, C.VOLUME_MOUNT),
                "files": 1,
                "bytes": os.path.getsize(path),
            }
        else:
            out["artifacts"][name] = {"path": os.path.relpath(path, C.VOLUME_MOUNT),
                                      "missing": True}
    out["total_bytes"] = sum(
        a.get("bytes", 0) for a in out["artifacts"].values()
    )
    return out


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=6 * _HOUR,
    cpu=8.0,
    memory=16384,
)
def push(repo_id: str, stats: dict, private: bool = False) -> dict:
    """Publish mel + manifests + vocab: what training from scratch needs."""
    import os

    from huggingface_hub import HfApi

    src = os.path.join(C.VOLUME_MOUNT, FEATURES)
    if not os.path.isdir(src):
        return {"error": f"{src} not found"}
    tree = {k: v for k, v in _tree(src).items() if not k.startswith("index/")}

    api = HfApi(token=_token())
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # A folder upload rather than per-file: it batches, resumes, and skips
    # blobs the repo already has, which matters at 300+ files and 25 GB.
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=src,
        path_in_repo=FEATURES,
        # The index is a rebuildable cache, so it is left behind rather than
        # bloating the download for everyone who clones this.
        ignore_patterns=["index/**"],
        commit_message="mel features, alignment manifests and unit vocabulary",
    )

    api.upload_file(
        path_or_fileobj=_card(repo_id, stats).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="dataset card",
    )

    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "files": len(tree),
        "bytes": sum(tree.values()),
        "carded": bool(stats.get("audio_hours")),
    }


@app.function(
    image=image,
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=6 * _HOUR,
    cpu=8.0,
    memory=16384,
)
def pull(repo_id: str) -> dict:
    """Fetch mel + manifests + vocab onto this workspace's volume."""
    import os
    import shutil

    from huggingface_hub import snapshot_download

    dest = os.path.join(C.VOLUME_MOUNT, FEATURES)
    # Straight into the mount rather than the HF cache: a cache copy would put
    # 24 GB on the volume twice.
    snap = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{FEATURES}/**"],
        local_dir=os.path.join(C.VOLUME_MOUNT, "_incoming"),
        token=_token(),
        max_workers=8,
    )
    staged = os.path.join(snap, FEATURES)
    for sub in ("mel", "manifest"):
        src_sub = os.path.join(staged, sub)
        dst_sub = os.path.join(dest, sub)
        if os.path.isdir(dst_sub):
            shutil.rmtree(dst_sub)
        os.makedirs(dest, exist_ok=True)
        shutil.move(src_sub, dst_sub)
    for f in os.listdir(staged):
        src_f = os.path.join(staged, f)
        if os.path.isfile(src_f):
            shutil.move(src_f, os.path.join(dest, f))

    shutil.rmtree(snap, ignore_errors=True)
    volume.commit()

    tree = _tree(dest)
    return {
        "features_dir": dest,
        "files": len(tree),
        "bytes": sum(tree.values()),
        "mel_shards": sum(1 for k in tree if k.startswith("mel/")),
        "manifests": sum(1 for k in tree if k.startswith("manifest/")),
        # Expected to be 0 on a fresh volume: the index hops in separately, or
        # the first training run rebuilds it.
        "index_files": sum(1 for k in tree if k.startswith("index/")),
    }


@app.local_entrypoint()
def main(repo_id: str, direction: str = "push", private: bool = False) -> None:
    """`--direction push` in the source workspace, `pull` in the target."""
    import json

    if direction == "push":
        # The card's numbers come from `modal run modal_app/train.py::inspect`,
        # kept in the repo so the published card cannot drift from what the
        # released model was trained on.
        import os

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "docs", "store_stats.json"), encoding="utf-8") as fh:
            stats = json.load(fh)
        out = push.remote(repo_id=repo_id, stats=stats, private=private)
    else:
        out = pull.remote(repo_id=repo_id)
    print(json.dumps(out, indent=2))
