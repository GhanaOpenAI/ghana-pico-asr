"""Receptive-field sweep: size the temporal context empirically.

The receptive field is how much audio each frame's prediction sees. It is not a
labelling choice — labels are always one per 10 ms frame — but it is the one
architectural number worth measuring rather than guessing.

Trains a short run per dilation stack on identical data and prints the
comparison, including the two contrasts the aligner cannot itself represent
(ɛ/e and ɔ/o), which are the honest test of whether the CNN is learning from
audio rather than parroting the alignment.

    export MODAL_PROFILE=ghana-nlp
    modal run modal_app/sweep.py
    modal run modal_app/sweep.py --stacks "1,2|1,2,4|1,2,4,8" --epochs 3
"""

import json

from modal_app.common import VOLUMES, app, image, volume
from ghana_pico_asr import config as C

image = image.add_local_python_source("modal_app")


@app.function(image=image, gpu="a10g", volumes=VOLUMES, timeout=60 * 60 * 4)
def train_one(ccfg_kwargs: dict, tcfg_kwargs: dict) -> dict:
    from ghana_pico_asr.trainer import train

    summary = train(
        C.VOLUME_MOUNT,
        C.ChunkConfig(**ccfg_kwargs),
        C.TrainConfig(**tcfg_kwargs),
        device="cuda",
    )
    volume.commit()
    return {
        "dilations": list(tcfg_kwargs["dilations"]),
        "receptive_field_ms": summary["receptive_field_ms"],
        "n_params": summary["n_params"],
        "n_train": summary["chunks"]["train"],
        "test": summary["test"],
        "contrasts": summary["test_contrasts"],
    }


@app.local_entrypoint()
def main(
    stacks: str = "1,2|1,2,4|1,2,4,8|1,2,4,8,16",
    epochs: int = 3,
    max_train_chunks: int = 60_000,
    splits: str = "tts,asr",
):
    split_tuple = tuple(s.strip() for s in splits.split(",") if s.strip())
    dil_stacks = [
        tuple(int(d) for d in stack.split(",") if d.strip())
        for stack in stacks.split("|")
        if stack.strip()
    ]

    jobs = [
        (
            {"splits": split_tuple},
            {
                "dilations": dil,
                "epochs": epochs,
                "max_train_chunks": max_train_chunks,
                "run_name": "sweep_d" + "-".join(str(d) for d in dil),
            },
        )
        for dil in dil_stacks
    ]

    results = list(train_one.starmap(jobs, order_outputs=True, return_exceptions=True))

    print("\n=== receptive-field sweep ===")
    header = (
        f"{'context':>10} {'dilations':>16} {'params':>10} "
        f"{'frame acc':>10} {'macro F1':>9} {'unit ER':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        t = r["test"]
        dil = ",".join(str(d) for d in r["dilations"])
        print(
            f"{r['receptive_field_ms']:>7} ms {dil:>16} {r['n_params']:>10,} "
            f"{t['frame_acc']:>10.4f} {t['macro_f1']:>9.4f} {t['unit_error_rate']:>8.4f}"
        )

    print("\n=== contrasts the aligner cannot represent ===")
    for r in results:
        pairs = r["contrasts"]
        bits = []
        for key in ("ɛ/e", "ɔ/o"):
            if key in pairs:
                a, b = key.split("/")
                bits.append(f"{key} {pairs[key][f'recall_{a}']}/{pairs[key][f'recall_{b}']}")
        print(f"  {r['receptive_field_ms']:>4} ms  " + "  ".join(bits))

    best = min(results, key=lambda r: r["test"]["unit_error_rate"])
    print(
        f"\nbest unit error rate: {best['receptive_field_ms']} ms context "
        f"({best['test']['unit_error_rate']:.4f})"
    )
    with open("sweep_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("full results -> sweep_results.json")
