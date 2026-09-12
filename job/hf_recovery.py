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
    ap.add_argument("--extend-tokenizer", action="store_true",
                    help="add any Twi characters the tokeniser cannot represent, "
                         "and train the resized embeddings alongside the adapters")
    ap.add_argument("--clean-text-repo", default=None,
                    help="HF text dataset for clean pairs; empty string reuses "
                         "the training pairs' own reference_units instead")
    ap.add_argument("--clean-ratio", type=float, default=0.0,
                    help="extra pairs built from reference_units, as a fraction "
                         "of the real pairs. Teaches restoration without errors "
                         "to correct; too high and the model learns to trust "
                         "its input instead of fixing it")
    ap.add_argument("--spaced-input", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pairs-repo", default=None)

    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--task-prefix", default=None,
                    help="prepended to every input; defaults to '' for NLLB and "
                         "'restore twi: ' for the T5 family, which has no "
                         "language conditioning to tell it what to do")
    ap.add_argument("--push-to", default=None, help="HF model repo to publish to")
    ap.add_argument("--eval-only", default=None, metavar="ADAPTER_DIR",
                    help="skip training; load this saved adapter and score it")
    ap.add_argument("--eval-pairs", default=None, metavar="JSONL",
                    help="score against these pairs instead of the held-out "
                         "training split. Use the Waxal set: its targets are "
                         "human transcripts, where the training corpora are "
                         "63%% machine transcript, so this measures correctness "
                         "rather than agreement with another recogniser")
    return ap


#: Pinned, not floored. `transformers>=4.44` resolves to whatever is newest at
#: job time, and a release that drops a Trainer argument then fails a run that
#: worked yesterday — which is how `warmup_ratio` disappeared mid-session.
#: These are the versions the code is verified against locally.
#:
#: The runtime is pinned too, by image tag: these versions need torch >= 2.7
#: (peft reaches for `torch.float8_e8m0fnu`), so the launcher asks for a
#: torch-2.10 image. Pinning libraries onto an unpinned runtime is half a pin.
DEPS = (
    "transformers==5.7.0",
    "peft==0.19.0",
    "accelerate==1.12.0",
    "sentencepiece==0.2.0",
    "pyarrow==23.0.1",
    "numpy==1.26.4",
    "huggingface_hub==1.23.0",
    # mT5's tokeniser is stored as a tiktoken file in transformers 5.x, and
    # protobuf is needed to convert sentencepiece models. Neither is pulled in
    # by transformers itself.
    "tiktoken==0.9.0",
    "protobuf==6.33.5",
)


def ensure_deps(packages: tuple[str, ...] = DEPS) -> None:
    """Install the pinned dependency set.

    Always invoked rather than skipped when the imports happen to resolve: an
    image that already carries a different version of transformers would
    otherwise be used silently, and the whole point of pinning is that the job
    runs the versions the code was tested against. pip is a no-op in seconds
    when the pins are already satisfied.

    Doing this from inside Python rather than as `sh -c "pip install ... && python ..."`
    keeps the job command a plain argv, with no shell quoting to get wrong.
    """
    import subprocess

    print(f"[deps] ensuring {len(packages)} pinned packages", flush=True)
    # --break-system-packages: newer base images mark their Python as
    # externally managed (PEP 668) and pip refuses to touch it. The container
    # is disposable, so there is no system to protect.
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--break-system-packages", *packages],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Printed rather than swallowed: `pip install -q` hid an
        # externally-managed-environment error behind a bare CalledProcessError,
        # which cost a scheduling round trip to diagnose.
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], flush=True)
        raise SystemExit(f"[deps] pip failed with status {proc.returncode}")
    import importlib.metadata as md

    print(
        "[deps] "
        + " ".join(
            f"{p.split('==')[0]}={md.version(p.split('==')[0].replace('_', '-'))}"
            for p in packages
        ),
        flush=True,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ensure_deps()

    import numpy as np
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from ghana_pico_asr.recovery.causal import build_example
    from ghana_pico_asr.recovery.causal import collate as causal_collate
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
            clean_ratio=args.clean_ratio,
            **({} if args.clean_text_repo is None
               else {'clean_text_repo': args.clean_text_repo or None}),
        ),
        limit=args.limit,
        spaced=args.spaced_input,
        cache_dir=os.path.join(WORK, "recovery", "cache"),
    )
    splits = split_pairs(rows)
    report["splits"] = {k: len(v) for k, v in splits.items()}

    if args.eval_pairs:
        # An external, human-transcribed test set replaces the held-out split.
        from ghana_pico_asr.recovery.data import format_source

        ext = []
        with open(args.eval_pairs, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                r["source_text"] = format_source(r["units"], spaced=args.spaced_input)
                r["target_text"] = r["text"].strip()
                ext.append(r)
        splits["test"] = ext
        report["external_test"] = {"path": args.eval_pairs, "n": len(ext)}
    print(f"[data] {json.dumps(report)} in {time.time() - t0:.0f}s", flush=True)

    # ---- model --------------------------------------------------------- #
    # NLLB conditions on language codes; the T5 family has none and uses a
    # task prefix instead. Detected from the tokeniser rather than the model
    # name, so a new checkpoint of either family just works.
    probe = AutoTokenizer.from_pretrained(args.base_model)
    is_nllb = args.lang_code in probe.get_vocab()
    # Encoder-decoder or decoder-only, read from the config rather than the
    # model name: the two need different data layout, loss masking, LoRA
    # targets and generation.
    is_causal = not getattr(
        AutoConfig.from_pretrained(args.base_model), "is_encoder_decoder", False
    )
    tok = (
        AutoTokenizer.from_pretrained(
            args.base_model, src_lang=args.lang_code, tgt_lang=args.lang_code
        )
        if is_nllb
        else probe
    )
    family = "nllb" if is_nllb else ("causal" if is_causal else "t5")
    print(f"[model] {args.base_model} family={family}", flush=True)
    if is_causal and probe.pad_token_id is None:
        # Qwen ships no pad token; padding with EOS is standard, and the
        # attention mask keeps it out of the computation either way.
        probe.pad_token = probe.eos_token
    # use_safetensors: NLLB ships both pytorch_model.bin and model.safetensors,
    # and transformers 5.x refuses to torch.load a .bin on torch < 2.6
    # (CVE-2025-32434). Asking for safetensors sidesteps the version floor
    # instead of pinning the image to a newer torch than stage 1 runs on.
    # Some tokenisers cannot represent Twi at all: t5 emits UNK for Ɔ Ɛ ɔ ɛ and
    # round-trips "Ɔyɛ ne ho" to "y ne ho", silently deleting the two vowels the
    # grapheme inventory exists to carry. Adding them is four tokens.
    added = 0
    if args.extend_tokenizer:
        alphabet = set("abcdefghijklmnopqrstuvwxyzɛɔ")
        alphabet |= {c.upper() for c in alphabet}
        missing = sorted(
            c for c in alphabet
            if tok.unk_token_id is not None
            and tok.unk_token_id in tok(c, add_special_tokens=False)["input_ids"]
        )
        if missing:
            added = tok.add_tokens(missing)
            print(f"[tok] added {added} tokens: {' '.join(missing)}", flush=True)

    loader = AutoModelForCausalLM if is_causal else AutoModelForSeq2SeqLM
    model = loader.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )

    if added:
        model.resize_token_embeddings(len(tok))

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
            task_type="CAUSAL_LM" if is_causal else "SEQ_2_SEQ_LM",
            # Newly added tokens start with random embeddings, and LoRA does
            # not touch the embedding matrix — so without this they would stay
            # random for the whole run and the model could never read or write
            # ɛ and ɔ. Saving the embedding layer costs memory; not saving it
            # makes the extension pointless.
            modules_to_save=(["shared", "lm_head"] if added else None),
            # Every linear layer in either family's blocks; the two use
            # different names for the same places.
            target_modules=(
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
                if is_causal
                else ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                if is_nllb
                else ["q", "k", "v", "o", "wi", "wi_0", "wi_1", "wo"]
            ),
        )
        model = get_peft_model(model, cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[lora] r={args.lora_r} alpha={args.lora_alpha} "
              f"trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)",
              flush=True)

    # The target language token NLLB must open every generation with. It goes
    # on `generation_config` only: transformers 5.x rejects generation settings
    # left on `model.config`, treating them as a modified pretrained config.
    # T5 has no such token and must not have one forced.
    forced_bos = tok.convert_tokens_to_ids(args.lang_code) if is_nllb else None
    if forced_bos is not None:
        for owner in (model, getattr(model, "base_model", None)):
            gc = getattr(owner, "generation_config", None)
            if gc is not None:
                gc.forced_bos_token_id = forced_bos

    # No prefix for NLLB (its language codes say what to do) and none for a
    # causal model (causal.PROMPT already wraps the units in an instruction).
    # Adding one for causal would also desync training from inference, since
    # the scorer rebuilds its prompt from the unprefixed units.
    prefix = args.task_prefix if args.task_prefix is not None else (
        "" if (is_nllb or is_causal) else "restore twi: "
    )
    if prefix:
        print(f"[model] task prefix {prefix!r}", flush=True)
        for r in splits["train"] + splits["val"] + splits["test"]:
            # Keep the unprefixed units: `baseline_cer` scores the raw input
            # against the reference, and counting the prefix as errors would
            # hand the T5 family an inflated baseline and make the families
            # incomparable.
            r["raw_source"] = r["source_text"]
            r["source_text"] = prefix + r["source_text"]

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
            r = self.rows[i]
            if is_causal:
                # One stream, prompt masked out of the loss.
                return build_example(
                    tok, r["source_text"], r["target_text"],
                    max_source_len=args.max_source_len,
                    max_target_len=args.max_target_len,
                )
            enc = encode([r])
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
        data_collator=(
            (lambda feats: causal_collate(feats, tok.pad_token_id))
            if is_causal
            else DataCollatorForSeq2Seq(tok, model=model)
        ),
    )
    if args.eval_only:
        from peft import PeftModel

        print(f"[eval-only] loading adapter {args.eval_only}", flush=True)
        base = loader.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
        )
        model = PeftModel.from_pretrained(base, args.eval_only)
        if forced_bos is not None:
            for owner in (model, getattr(model, "base_model", None)):
                gc = getattr(owner, "generation_config", None)
                if gc is not None:
                    gc.forced_bos_token_id = forced_bos
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        trainer.model = model
    else:
        trainer.train()

    # ---- score on held-out pairs --------------------------------------- #
    from ghana_pico_asr.recovery.evaluate import score_causal, score_model

    scorer = score_causal if is_causal else score_model
    metrics = scorer(
        trainer.model, tok, splits["test"][: args.eval_n],
        args.lang_code if is_nllb else None,
        max_source_len=args.max_source_len, max_target_len=args.max_target_len,
    )
    print(f"[test] {json.dumps(metrics)}", flush=True)

    if not args.eval_only:
        model.save_pretrained(os.path.join(out_dir, "final"))
        tok.save_pretrained(os.path.join(out_dir, "final"))
    # Re-created immediately before the write: the bucket mount is object
    # storage, where a directory containing no files does not reliably persist
    # between its creation and a later open(). Losing a completed evaluation to
    # that would be absurd, so the result is also printed above regardless.
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({"data": report, "test": metrics, "args": vars(args)}, fh, indent=2)
    except OSError as exc:
        print(f"[warn] could not write metrics.json ({exc}); "
              "the [test] line above carries the same numbers", flush=True)

    if args.push_to:
        model.push_to_hub(args.push_to)
        tok.push_to_hub(args.push_to)
        print(f"pushed -> https://huggingface.co/{args.push_to}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
