"""End-to-end decode demo: audio in, grapheme units out.

Exercises the real inference path — whole utterance, one forward pass, no
segmentation and no text — against a trained checkpoint, and prints the
decoded units beside the reference for eyeballing.

    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/decode.py --run-name twi2dcnn --n 5          # waxal, held out
    modal run modal_app/decode.py --run-name twi2dcnn --split asr --n 4
    modal run modal_app/decode.py --run-name twi2dcnn --split tts --n 6
"""

import json

from modal_app.common import VOLUMES, app, hf_secret, image
from ghana_pico_asr import config as C

image = image.add_local_python_source("modal_app")


@app.function(image=image, gpu="a10g", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 30)
def decode(
    run_name: str, split: str, n: int, max_dur: float, sweep: bool = True
) -> list[dict]:
    import os

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ghana_pico_asr import features as F
    from ghana_pico_asr.languages import flatten, get_language
    from ghana_pico_asr.infer import UnitTagger
    from ghana_pico_asr.trainer import collapse, edit_distance

    ckpt = os.path.join(C.VOLUME_MOUNT, C.CKPT_DIR, run_name, "best.pt")
    tagger = UnitTagger(ckpt, device="cuda")
    lang = tagger.language

    # Checkpoints written before unit_median_frames existed need them supplied
    # so long runs can be re-split into geminates.
    if tagger.median_frames is None:
        import torch as _t

        from ghana_pico_asr import dataset as D

        cc = C.ChunkConfig(**_t.load(ckpt, map_location="cpu", weights_only=False)["chunk_config"])
        vocab = D.load_or_build_vocab(C.VOLUME_MOUNT, cc)
        tagger.median_frames = D.unit_median_frames(C.VOLUME_MOUNT, cc, vocab)
        print("[infer] computed unit_median_frames from the feature store")

    print(
        f"[infer] {len(tagger.vocab)} classes | context {tagger.receptive_field_ms} ms"
        f" | geminate splitting {'on' if tagger.median_frames else 'off'}"
    )

    if split == "tts":
        repo, fn, col = C.TTS_REPO, "data/train-00000-of-00004.parquet", C.TTS_TEXT_COL
    elif split == "asr":
        repo, fn, col = C.ASR_REPO, "data/train-00000-of-00114.parquet", C.ASR_TEXT_COL
    elif split == "waxal":
        # Dev slice: used to pick decode settings.
        repo, fn, col = C.EVAL_REPO, C.EVAL_FILE, C.EVAL_TEXT_COL
    elif split == "waxal-heldout":
        # Never used for tuning, so its UER is the honest number.
        repo, fn, col = C.EVAL_REPO, C.EVAL_FILE_HELDOUT, C.EVAL_TEXT_COL
    else:
        raise ValueError(f"unknown split {split!r}; use tts, asr or waxal")
    local = hf_hub_download(
        repo, fn, repo_type="dataset", cache_dir=f"{C.VOLUME_MOUNT}/{C.HF_CACHE_DIR}/hub"
    )

    out = []
    pf = pq.ParquetFile(local)
    for batch in pf.iter_batches(batch_size=4, columns=["audio", col]):
        for row in batch.to_pylist():
            if len(out) >= n:
                return out
            text = (row.get(col) or "").strip()
            if not text:
                continue
            wav = F.decode_audio(row["audio"])
            dur = len(wav) / C.SAMPLE_RATE
            # The health corpus is a flat 30 s per chunk, so this bound has to
            # admit it -- a 12 s cap silently matched nothing.
            if not (1.0 <= dur <= max_dur):
                continue

            probs = tagger.posteriors(wav)
            ref = flatten(lang.segment(text))
            idx = {u: i for i, u in enumerate(tagger.vocab)}
            ref_ids = [idx[u] for u in ref if u in idx]

            def _uer(**kw):
                us = tagger.units_from_posteriors(probs, **kw)
                h = [u.unit for u in us]
                return us, h, edit_distance([idx[u] for u in h if u in idx], ref_ids) / max(
                    len(ref_ids), 1
                )

            grid = {}
            if sweep:
                # Every model came out at ~0.88 length ratio, i.e. deleting, so
                # the grid has to reach into *less* aggressive territory:
                # min_frames below 4 and smoothing both below and above 9.
                for sm in (1, 3, 5, 7, 9, 11, 13):
                    for mf in (1, 2, 3, 4, 5, 6):
                        _, h, u = _uer(smooth_frames=sm, min_frames=mf)
                        grid[f"sm{sm:02d}_mf{mf}"] = (round(u, 4), len(h))
            units, hyp, uer = _uer()
            _, _, uer_raw = _uer(smooth_frames=1, min_frames=2)
            out.append(
                {
                    "text": text,
                    "duration_s": round(dur, 2),
                    "reference": " ".join(ref),
                    "decoded": " ".join(hyp),
                    "n_ref": len(ref),
                    "n_hyp": len(hyp),
                    "unit_error_rate": round(uer, 3),
                    "uer_unsmoothed": round(uer_raw, 3),
                    "grid": grid,
                    "mean_confidence": round(float(np.mean([u.confidence for u in units])), 3)
                    if units
                    else None,
                    "frames": int(probs.shape[0]),
                }
            )
    return out


@app.local_entrypoint()
def main(
    run_name: str = "twi2dcnn",
    split: str = "waxal",
    n: int = 5,
    max_dur: float = 31.0,
    show_units: int = 70,
    sweep: bool = False,
):
    results = decode.remote(run_name, split, n, max_dur, sweep)

    def clip(seq: str) -> str:
        parts = seq.split()
        if len(parts) <= show_units:
            return seq
        return " ".join(parts[:show_units]) + f"  ... (+{len(parts) - show_units} more)"

    for i, r in enumerate(results):
        print("=" * 78)
        print(f"[{i}] {r['duration_s']}s, UER {r['unit_error_rate']} "
              f"(unsmoothed {r['uer_unsmoothed']}), "
              f"conf {r['mean_confidence']}  ({r['n_ref']} ref -> {r['n_hyp']} decoded)")
        print(f"  text : {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
        print(f"  ref  : {clip(r['reference'])}")
        print(f"  hyp  : {clip(r['decoded'])}")
    if results:
        n = len(results)
        mean = sum(r["unit_error_rate"] for r in results) / n
        plain = sum(r["uer_unsmoothed"] for r in results) / n
        print("=" * 78)
        print(f"mean utterance-level UER over {n} utterances: {mean:.3f}")
        print(f"  unsmoothed (smooth=1, min_frames=2):        {plain:.3f}")
        if not results[0]["grid"]:
            return
        ref_mean = sum(r["n_ref"] for r in results) / n
        print(f"\n=== decode knob sweep ===   reference mean {ref_mean:.0f} units")
        print(f"  {'setting':>12} {'UER':>8} {'units':>7} {'ratio':>7}")
        rows = []
        for k in results[0]["grid"]:
            u = sum(r["grid"][k][0] for r in results) / n
            c = sum(r["grid"][k][1] for r in results) / n
            rows.append((u, k, c, c / max(ref_mean, 1)))
        for u, k, c, ratio in sorted(rows)[:18]:
            flag = "  <- ratio near 1.0" if 0.95 <= ratio <= 1.05 else ""
            print(f"  {k:>12} {u:>8.4f} {c:>7.0f} {ratio:>7.3f}{flag}")
        best = min(rows)
        print(f"\n  best UER: {best[1]} -> {best[0]:.4f} (ratio {best[3]:.3f})")
    with open("decode_samples.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
