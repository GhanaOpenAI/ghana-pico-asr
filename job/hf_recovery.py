"""Fine-tune NLLB-200 to turn grapheme units into Twi text. Runs on HF Jobs.

Stage 1 emits sounds; this recovers words. NLLB rather than a T5: `aka_Latn`
and `twi_Latn` are both among its 202 languages, its tokeniser round-trips
`ɛ`/`ɔ` losslessly and spells `adwuma` as one token, so the Twi is already
there to borrow.

LoRA rather than a full fine-tune, for a reason that is about data rather than
compute: 63% of the published pairs carry targets written by Google STT or
Gemini, and a recovery model learns to write whatever its targets say. Updating
600M weights on those would bake another recogniser's mistakes into the model;
adapters leave NLLB's own Twi underneath. The rank is high (32) and every
linear layer is adapted, because the input — an unsegmented, ~35%-corrupted
grapheme stream — is far from anything NLLB saw in pretraining.

    python scripts/launch_recovery_job.py --smoke
    python scripts/launch_recovery_job.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

WORK = os.environ.get("PICO_WORK", "/work")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name", default="nllb-recovery")
    ap.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--lang-code", default="twi_Latn",
                    help="NLLB code for both sides; aka_Latn is the alternative")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-source-len", type=int, default=192)
    ap.add_argument("--max-target-len", type=int, default=160)

    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--full-finetune", action="store_true",
                    help="update all weights instead of adapters, for comparison")

    ap.add_argument("--max-uer", type=float, default=0.5)
    ap.add_argument("--machine-ratio", type=float, default=0.5)
    ap.add_argument("--min-units", type=int, default=8)
    ap.add_argument("--spaced-input", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pairs-repo", default=None)

    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--push-to", default=None, help="HF model repo to publish to")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import numpy as np
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from ghana_pico_asr.recovery.data import PairFilter, load_pairs, split_pairs

    print(f"[env] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[env] gpu {p.name} {p.total_memory / 1e9:.0f} GB", flush=True)

    # ---- data ---------------------------------------------------------- #
    t0 = time.time()
    rows, report = load_pairs(
        repo_id=args.pairs_repo,
        flt=PairFilter(
            max_uer=args.max_uer,
            min_units=args.min_units,
            machine_ratio=args.machine_ratio,
        ),
        limit=args.limit,
        spaced=args.spaced_input,
    )
    splits = split_pairs(rows)
    report["splits"] = {k: len(v) for k, v in splits.items()}
    print(f"[data] {json.dumps(report)} in {time.time() - t0:.0f}s", flush=True)

    # ---- model --------------------------------------------------------- #
    tok = AutoTokenizer.from_pretrained(
        args.base_model, src_lang=args.lang_code, tgt_lang=args.lang_code
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model

        # Every linear layer, not just attention: the encoder has to learn an
        # input distribution NLLB has never seen, which attention-only adapters
        # at rank 8 do not have the capacity for.
        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        )
        model = get_peft_model(model, cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[lora] r={args.lora_r} alpha={args.lora_alpha} "
              f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)",
              flush=True)

    forced_bos = tok.convert_tokens_to_ids(args.lang_code)
    model.config.forced_bos_token_id = forced_bos
    if hasattr(model, "generation_config"):
        model.generation_config.forced_bos_token_id = forced_bos

    def encode(batch_rows):
        model_inputs = tok(
            [r["source_text"] for r in batch_rows],
            text_target=[r["target_text"] for r in batch_rows],
            max_length=args.max_source_len,
            truncation=True,
        )
        # Targets get their own cap; the source is the longer of the two.
        labels = tok(
            text_target=[r["target_text"] for r in batch_rows],
            max_length=args.max_target_len,
            truncation=True,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    class Pairs(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            enc = encode([self.rows[i]])
            return {k: v[0] for k, v in enc.items()}

    train_ds, val_ds = Pairs(splits["train"]), Pairs(splits["val"][: args.eval_n])

    out_dir = os.path.join(WORK, "recovery", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    targs = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Every epoch is kept. On the recogniser, overwriting the best
        # checkpoint destroyed the epoch that turned out to matter.
        save_total_limit=None,
        predict_with_generate=False,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    trainer.train()

    # ---- score on held-out pairs --------------------------------------- #
    from ghana_pico_asr.recovery.evaluate import score_model

    metrics = score_model(
        model, tok, splits["test"][: args.eval_n], args.lang_code,
        max_source_len=args.max_source_len, max_target_len=args.max_target_len,
    )
    print(f"[test] {json.dumps(metrics)}", flush=True)

    model.save_pretrained(os.path.join(out_dir, "final"))
    tok.save_pretrained(os.path.join(out_dir, "final"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"data": report, "test": metrics, "args": vars(args)}, fh, indent=2)

    if args.push_to:
        model.push_to_hub(args.push_to)
        tok.push_to_hub(args.push_to)
        print(f"pushed -> https://huggingface.co/{args.push_to}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
