"""Stage 1 on Modal: forced-align both corpora and write the feature store.

    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/prepare.py                  # both corpora, 10k each
    modal run modal_app/prepare.py --smoke          # 1 shard/corpus, 150 utts
    modal run modal_app/prepare.py --splits tts     # one corpus only
    modal run modal_app/prepare.py --splits female,agric,multispk   # extra corpora
    modal run modal_app/prepare.py --splits kuma --smoke --smoke-utts 300  # quality probe
"""

import json

import modal

from modal_app.common import VOLUMES, app, hf_secret, image, volume
from ghana_pico_asr import config as C
from ghana_pico_asr import prepare as P

image = image.add_local_python_source("modal_app")


@app.function(
    image=image,
    gpu="a10g",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=60 * 60 * 3,
    # A crash inside libsndfile/ffmpeg is a SIGSEGV, which Python cannot catch,
    # so a single unreadable file takes the whole shard down. Retries give it
    # another go; the memory request is deliberate because some corpora hold
    # 30 s utterances and the default was too tight.
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    memory=16384,
    cpu=4.0,
    max_containers=20,
)
def align_shard(job: dict) -> dict:
    """One parquet shard -> one mel blob + one unit manifest on the volume."""
    import os
    import traceback

    # The `huggingface` secret may expose either spelling.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    try:
        result = P.process_shard(
            repo_id=job["repo_id"],
            split=job["split"],
            shard=job["shard"],
            n_shards=job["n_shards"],
            text_col=job["text_col"],
            out_root=C.VOLUME_MOUNT,
            max_utts=job["max_utts"],
            device="cuda",
            hf_cache=f"{C.VOLUME_MOUNT}/{C.HF_CACHE_DIR}/hub",
            overwrite=job.get("overwrite", False),
            decode_backend=job.get("decode_backend", "soundfile"),
        )
    except Exception as exc:  # noqa: BLE001 - one bad shard must not sink the run
        traceback.print_exc()
        return {
            "split": job["split"],
            "shard": job["shard"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    volume.commit()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def build_jobs(
    splits: tuple[str, ...], smoke: bool, overwrite: bool, smoke_utts: int = 150
) -> list[dict]:
    jobs: list[dict] = []

    if "tts" in splits:
        shards = list(range(C.TTS_N_SHARDS))
        if smoke:
            shards = shards[:1]
        per = smoke_utts if smoke else -(-C.TTS_TARGET_UTTS // len(shards))
        jobs += [
            {
                "repo_id": C.TTS_REPO,
                "split": "tts",
                "shard": s,
                "n_shards": C.TTS_N_SHARDS,
                "text_col": C.TTS_TEXT_COL,
                "max_utts": per,
                "overwrite": overwrite,
            }
            for s in shards
        ]

    if "kuma" in splits:
        shards = sorted(
            set(
                __import__("numpy")
                .linspace(0, C.KUMA_N_SHARDS - 1, C.KUMA_N_SHARDS_USED)
                .round()
                .astype(int)
                .tolist()
            )
        )
        if smoke:
            shards = shards[:3]
        per = smoke_utts if smoke else -(-C.KUMA_TARGET_UTTS // len(shards))
        jobs += [
            {
                "repo_id": C.KUMA_REPO,
                "split": "kuma",
                "shard": s,
                "n_shards": C.KUMA_N_SHARDS,
                "text_col": C.KUMA_TEXT_COL,
                "max_utts": per,
                "overwrite": overwrite,
            }
            for s in shards
        ]

    # The later corpora are all plain audio+text parquet, so one loop covers
    # them; only the repo, shard count and text column differ.
    for name, spec in C.EXTRA_SOURCES.items():
        if name not in splits:
            continue
        shards = list(range(spec["n_shards"]))
        if smoke:
            shards = shards[:1]
        per = smoke_utts if smoke else -(-spec["rows"] * 2 // len(shards))
        jobs += [
            {
                "repo_id": spec["repo"],
                "split": name,
                "shard": s,
                "n_shards": spec["n_shards"],
                "text_col": spec["text_col"],
                "max_utts": per,
                "overwrite": overwrite,
            }
            for s in shards
        ]

    if "asr" in splits:
        shards = P.select_asr_shards()
        if smoke:
            shards = shards[:1]
        per = smoke_utts if smoke else -(-C.ASR_TARGET_UTTS // len(shards))
        jobs += [
            {
                "repo_id": C.ASR_REPO,
                "split": "asr",
                "shard": s,
                "n_shards": C.ASR_N_SHARDS,
                "text_col": C.ASR_TEXT_COL,
                "max_utts": per,
                "overwrite": overwrite,
            }
            for s in shards
        ]

    return jobs


@app.local_entrypoint()
def main(
    splits: str = "tts,asr",
    smoke: bool = False,
    overwrite: bool = False,
    smoke_utts: int = 150,
):
    wanted = tuple(s.strip() for s in splits.split(",") if s.strip())
    jobs = build_jobs(wanted, smoke, overwrite, smoke_utts)
    print(f"dispatching {len(jobs)} shard jobs ({'smoke' if smoke else 'full'})")

    raw = list(align_shard.map(jobs, order_outputs=False, return_exceptions=True))

    # A container killed by a native crash (SIGSEGV in libsndfile/ffmpeg, which
    # Python cannot catch) comes back as an exception rather than a result.
    # Without return_exceptions the first such shard aborts every remaining
    # one, which cost 11 shards on a 48-shard run.
    results = [r for r in raw if isinstance(r, dict)]
    crashed = [r for r in raw if not isinstance(r, dict)]

    ok = [r for r in results if r.get("status") in ("ok", "cached")]
    bad = [r for r in results if r.get("status") == "error"]
    print("\n=== per-shard ===")
    for r in sorted(results, key=lambda r: (r["split"], r["shard"])):
        print(json.dumps(r, ensure_ascii=False))

    print("\n=== totals ===")
    for split in wanted:
        rows = [r for r in ok if r["split"] == split and r.get("status") == "ok"]
        if not rows:
            continue
        print(
            json.dumps(
                {
                    "split": split,
                    "shards": len(rows),
                    "utts": sum(r["n_utts"] for r in rows),
                    "units": sum(r["n_units"] for r in rows),
                    "audio_hours": round(sum(r["audio_hours"] for r in rows), 2),
                    "mel_frames": sum(r["mel_frames"] for r in rows),
                    "mel_mb": round(sum(r["mel_mb"] for r in rows), 1),
                    "mean_score": round(
                        sum(r["mean_score"] * r["n_utts"] for r in rows)
                        / max(1, sum(r["n_utts"] for r in rows)),
                        4,
                    ),
                }
            )
        )
    if bad:
        print(f"\n{len(bad)} shard(s) returned an error:")
        for r in bad:
            print(f"  {r['split']}/{r['shard']}: {r['error']}")
    if crashed:
        print(f"\n{len(crashed)} shard(s) crashed in native code (no result returned):")
        for e in crashed[:5]:
            print(f"  {type(e).__name__}: {str(e)[:140]}")
        print("  Re-run the same command: completed shards are cached, so only")
        print("  these are retried.")
