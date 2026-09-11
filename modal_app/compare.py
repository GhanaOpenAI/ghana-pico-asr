"""Compare every trained model on the held-out Waxal set, identically.

This is the honest test: whole utterances, **human** transcripts, a corpus none
of the models trained on, and no forced aligner in the reference. Every model
sees the same utterances with the same decode settings, so the numbers are
directly comparable — unlike the validation/test UERs, which are measured
against the aligner's own labels and saturate at its noise floor.

    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/compare.py --runs twi2dcnn,big-cap,big-ctx,wide-ctx --n 100
"""

import json

from modal_app.common import VOLUMES, app, hf_secret, image, volume
from ghana_pico_asr import config as C

image = image.add_local_python_source("modal_app")


@app.function(image=image, volumes=VOLUMES, timeout=15 * 60, cpu=2.0, memory=8192)
def snapshot_final_epoch(run_name: str, dest_name: str) -> dict:
    """Write `<dest>/best.pt` from `<run>/last.pt`'s weights, so a rejected
    final epoch can be scored on the held-out set like any other run."""
    import os

    import torch

    from ghana_pico_asr.release import releasable_from_resume

    src = os.path.join(C.VOLUME_MOUNT, C.CKPT_DIR, run_name)
    best_p, last_p = os.path.join(src, "best.pt"), os.path.join(src, "last.pt")
    for p in (best_p, last_p):
        if not os.path.exists(p):
            return {"error": f"{p} not found"}

    best = torch.load(best_p, map_location="cpu", weights_only=False)
    last = torch.load(last_p, map_location="cpu", weights_only=False)
    out = releasable_from_resume(best, last)

    dest = os.path.join(C.VOLUME_MOUNT, C.CKPT_DIR, dest_name)
    os.makedirs(dest, exist_ok=True)
    torch.save(out, os.path.join(dest, "best.pt"))
    volume.commit()
    return {
        "dest": dest_name,
        "selected_epoch": best.get("epoch"),
        "promoted_epoch": out.get("epoch"),
        "val": out.get("val"),
    }


@app.function(image=image, gpu="a10g", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 45)
def evaluate_run(run_name: str, n: int) -> dict:
    import os

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from ghana_pico_asr import features as F
    from ghana_pico_asr.languages import flatten, get_language
    from ghana_pico_asr.infer import UnitTagger
    from ghana_pico_asr.trainer import edit_distance

    ckpt = os.path.join(C.VOLUME_MOUNT, C.CKPT_DIR, run_name, "best.pt")
    if not os.path.exists(ckpt):
        return {"run": run_name, "error": "no checkpoint"}
    tagger = UnitTagger(ckpt, device="cuda")
    lang = tagger.language
    idx = {u: i for i, u in enumerate(tagger.vocab)}

    local = hf_hub_download(
        C.EVAL_REPO, C.EVAL_FILE_HELDOUT, repo_type="dataset",
        cache_dir=f"{C.VOLUME_MOUNT}/{C.HF_CACHE_DIR}/hub",
    )

    tot_err = tot_ref = tot_hyp = 0
    per_utt = []
    for batch in pq.ParquetFile(local).iter_batches(batch_size=4, columns=["audio", "text"]):
        if len(per_utt) >= n:
            break
        for row in batch.to_pylist():
            if len(per_utt) >= n:
                break
            text = (row.get("text") or "").strip()
            if not text:
                continue
            wav = F.decode_audio(row["audio"])
            ref = [idx[u] for u in flatten(lang.segment(text)) if u in idx]
            if len(ref) < 5:
                continue
            units = tagger.units(wav)
            hyp = [idx[u.unit] for u in units if u.unit in idx]
            e = edit_distance(hyp, ref)
            tot_err += e
            tot_ref += len(ref)
            tot_hyp += len(hyp)
            per_utt.append(e / len(ref))

    return {
        "run": run_name,
        "context_ms": tagger.receptive_field_ms,
        "n_params": sum(p.numel() for p in tagger.model.parameters()),
        "n_utts": len(per_utt),
        # Corpus-level: total edits / total reference units. The number to quote.
        "uer": round(tot_err / max(tot_ref, 1), 4),
        # Mean of per-utterance rates; short utterances weigh equally here.
        "uer_macro": round(float(np.mean(per_utt)), 4) if per_utt else None,
        "ref_units": tot_ref,
        "hyp_units": tot_hyp,
        "length_ratio": round(tot_hyp / max(tot_ref, 1), 3),
    }


@app.local_entrypoint()
def main(runs: str = "twi2dcnn,big-cap,big-ctx,wide-ctx", n: int = 100):
    names = [r.strip() for r in runs.split(",") if r.strip()]
    results = list(evaluate_run.starmap([(r, n) for r in names], order_outputs=True, return_exceptions=True))

    ok = [r for r in results if "error" not in r]
    print(f"\n=== held-out Waxal Asante Twi (shard 1, human transcripts), {n} utts ===")
    hdr = f"{'run':>11} {'params':>10} {'ctx':>7} {'UER':>7} {'UER/utt':>8} {'len ratio':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(ok, key=lambda r: r["uer"]):
        print(f"{r['run']:>11} {r['n_params']:>10,} {r['context_ms']:>5} ms "
              f"{r['uer']:>7.4f} {r['uer_macro']:>8.4f} {r['length_ratio']:>10.3f}")
    for r in results:
        if "error" in r:
            print(f"  {r['run']}: {r['error']}")
    print("\nlength ratio = decoded units / reference units (1.0 is ideal;"
          " >1 over-generates, <1 deletes)")
    with open("waxal_comparison.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
