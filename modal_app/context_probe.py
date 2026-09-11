"""How much does the model rely on surrounding audio vs the segment itself?

A wide receptive field lets a frame classifier infer a unit from the sound
*around* it — co-articulation, and in Akan also ATR vowel harmony — rather
than from that segment's own acoustics. That is helpful for recognition and
harmful for pronunciation scoring, so it is worth measuring rather than
assuming.

The probe: take the aligner's unit spans, and classify each one twice.

  full     the whole utterance is visible (normal inference)
  isolated everything outside the unit's own span is zeroed, so only the
           segment's own acoustics remain

Agreement between the two says the decision was acoustic. A large drop says it
was contextual. `margin_ms` sweeps a middle ground: keep that much audio each
side of the span and watch how fast accuracy recovers.

    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/context_probe.py --run-name big-ctx --n 60
"""

import json

from modal_app.common import VOLUMES, app, hf_secret, image
from ghana_pico_asr import config as C

image = image.add_local_python_source("modal_app")

MARGINS = [0, 20, 40, 80, 160, 320]  # ms kept either side of the span


@app.function(image=image, gpu="a10g", volumes=VOLUMES, secrets=[hf_secret], timeout=60 * 40)
def probe(run_name: str, n: int) -> dict:
    import os
    from collections import Counter, defaultdict

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download

    from ghana_pico_asr import features as F
    from ghana_pico_asr.align import align_utterance, load_aligner
    from ghana_pico_asr.infer import UnitTagger

    tagger = UnitTagger(
        os.path.join(C.VOLUME_MOUNT, C.CKPT_DIR, run_name, "best.pt"), device="cuda"
    )
    lang = tagger.language
    cls_of = {u: i for i, u in enumerate(tagger.vocab)}
    print(f"[probe] {run_name}: context {tagger.receptive_field_ms} ms", flush=True)

    # The aligner gives ground-truth spans on this held-out audio.
    model, tok = load_aligner(device="cuda")
    logmel = F.LogMel(device="cuda")
    local = hf_hub_download(
        C.EVAL_REPO, C.EVAL_FILE_HELDOUT, repo_type="dataset",
        cache_dir=f"{C.VOLUME_MOUNT}/{C.HF_CACHE_DIR}/hub",
    )

    hits = {m: Counter() for m in MARGINS}
    hits["full"] = Counter()
    totals: Counter = Counter()
    per_unit = {m: defaultdict(lambda: [0, 0]) for m in MARGINS + ["full"]}
    done = 0

    for batch in pq.ParquetFile(local).iter_batches(batch_size=2, columns=["audio", "text"]):
        if done >= n:
            break
        for row in batch.to_pylist():
            if done >= n:
                break
            text = (row.get("text") or "").strip()
            if not text:
                continue
            wav = F.decode_audio(row["audio"])
            try:
                spans = align_utterance(model, tok, wav, text)
            except Exception:
                continue

            mel = logmel(torch.from_numpy(wav))  # [T, n_mels]
            T = mel.shape[0]
            norm = ((mel - tagger.mean) / tagger.std).T  # [n_mels, T]

            # One full-utterance pass gives the "context available" answer.
            with torch.inference_mode():
                full_pred = (
                    tagger.model(norm.unsqueeze(0).unsqueeze(0)).argmax(1)[0].cpu().numpy()
                )

            keep = [
                (s, int(round(sp.start * 100)), int(round(sp.end * 100)))
                for s, sp in enumerate(spans)
                if sp.unit in cls_of and sp.score >= -1.0
            ]
            if not keep:
                continue

            for _, a, b in keep:
                a, b = max(0, a), min(T, b)
                if b - a < 2:
                    continue
                unit = spans[_].unit
                gold = cls_of[unit]
                mid = (a + b) // 2
                totals[unit] += 1

                pf = int(full_pred[mid]) == gold
                hits["full"][unit] += pf
                per_unit["full"][unit][0] += pf
                per_unit["full"][unit][1] += 1

                for m in MARGINS:
                    pad = m // C.FRAME_MS
                    lo, hi = max(0, a - pad), min(T, b + pad)
                    masked = torch.zeros_like(norm)
                    masked[:, lo:hi] = norm[:, lo:hi]
                    with torch.inference_mode():
                        p = tagger.model(masked.unsqueeze(0).unsqueeze(0)).argmax(1)[0, mid]
                    ok = int(p) == gold
                    hits[m][unit] += ok
                    per_unit[m][unit][0] += ok
                    per_unit[m][unit][1] += 1
            done += 1

    n_units = sum(totals.values())

    def acc(key):
        return sum(hits[key].values()) / max(n_units, 1)

    # Contrast pairs where harmony (context) could substitute for acoustics.
    pairs = [("ɛ", "e"), ("ɔ", "o")]
    contrast = {}
    for a, b in pairs:
        for key in ["full", 0, 80, 320]:
            ta = per_unit[key][a]
            tb = per_unit[key][b]
            tot = ta[1] + tb[1]
            if tot:
                contrast.setdefault(f"{a}/{b}", {})[str(key)] = round(
                    (ta[0] + tb[0]) / tot, 4
                )

    return {
        "run": run_name,
        "context_ms": tagger.receptive_field_ms,
        "n_utts": done,
        "n_units": n_units,
        "accuracy": {"full": round(acc("full"), 4), **{f"{m}ms": round(acc(m), 4) for m in MARGINS}},
        "contrast_accuracy": contrast,
        "per_unit_full_vs_isolated": {
            u: {
                "n": per_unit["full"][u][1],
                "full": round(per_unit["full"][u][0] / max(per_unit["full"][u][1], 1), 3),
                "isolated": round(per_unit[0][u][0] / max(per_unit[0][u][1], 1), 3),
            }
            for u in sorted(totals, key=lambda x: -totals[x])
            if per_unit["full"][u][1] >= 20
        },
    }


@app.local_entrypoint()
def main(run_name: str = "big-ctx", n: int = 60):
    r = probe.remote(run_name, n)
    print(f"\n=== context probe: {r['run']} ({r['context_ms']} ms context) ===")
    print(f"{r['n_units']:,} aligned units over {r['n_utts']} held-out utterances\n")
    full = r["accuracy"]["full"]
    print(f"  {'audio visible':>22}  {'centre-frame acc':>16}  {'% of full':>10}")
    print(f"  {'whole utterance':>22}  {full:>16.4f}  {'100.0':>10}")
    for m in MARGINS:
        a = r["accuracy"][f"{m}ms"]
        label = "span only" if m == 0 else f"span +/- {m} ms"
        print(f"  {label:>22}  {a:>16.4f}  {100 * a / full:>9.1f}%")
    print("\n  accuracy on the harmony-sensitive pairs:")
    for k, v in r["contrast_accuracy"].items():
        print(f"    {k}: " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
    with open(f"context_probe_{run_name}.json", "w", encoding="utf-8") as fh:
        json.dump(r, fh, ensure_ascii=False, indent=2)
    print(f"\nfull results -> context_probe_{run_name}.json")
