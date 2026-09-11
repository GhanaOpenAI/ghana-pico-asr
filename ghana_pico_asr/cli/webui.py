"""Browser UI: record or upload audio, see the unit sequence and timings."""

from __future__ import annotations

import sys

from ._common import resolve_checkpoint


def add_args(ap) -> None:
    ap.add_argument("-c", "--checkpoint", default=None, help="checkpoint file or directory")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument(
        "--share", action="store_true", help="create a public gradio link (use with care)"
    )
    ap.add_argument("--device", default=None)


def run(args) -> int:
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("the web UI needs:  pip install gradio") from None

    import numpy as np

    from .. import config as C
    from ..infer import UnitTagger

    ckpt = resolve_checkpoint(args.checkpoint)
    tagger = UnitTagger(ckpt, device=args.device)
    print(f"[pico] loaded {ckpt} ({tagger.language.name})", file=sys.stderr)

    def transcribe(audio, smooth, min_frames, min_conf, keep_sil, show_timings):
        if audio is None:
            return "", "", None
        sr, wav = audio
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # gradio hands back int16 for recordings
        peak = float(np.abs(wav).max() or 1.0)
        if peak > 1.5:
            wav = wav / 32768.0
        if sr != C.SAMPLE_RATE:
            import torch
            import torchaudio.functional as AF

            wav = AF.resample(torch.from_numpy(wav), sr, C.SAMPLE_RATE).numpy()

        units = tagger.units(
            np.ascontiguousarray(wav, dtype=np.float32),
            smooth_frames=int(smooth),
            min_frames=int(min_frames),
            min_confidence=float(min_conf),
            drop_silence=not keep_sil,
        )
        seq = " ".join(u.unit for u in units)
        conf = (
            f"{len(units)} units | mean confidence "
            f"{sum(u.confidence for u in units) / len(units):.3f}"
            if units
            else "no units decoded"
        )
        table = (
            [[u.unit, round(u.start, 3), round(u.end, 3), round(u.confidence, 3)] for u in units]
            if show_timings
            else None
        )
        return seq, conf, table

    with gr.Blocks(title=f"{C.PROJECT} — {tagger.language.name}") as demo:
        gr.Markdown(
            f"# {C.PROJECT}\n"
            f"**{tagger.language.name}** grapheme-unit recogniser — "
            f"{len(tagger.vocab)} classes, {tagger.receptive_field_ms} ms context.\n\n"
            "Output is a sequence of grapheme units, not a sentence. A separate "
            "text-recovery model turns these into words."
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
                with gr.Accordion("Decoding options", open=False):
                    smooth = gr.Slider(1, 21, value=7, step=2, label="Posterior smoothing (frames)")
                    minf = gr.Slider(1, 12, value=4, step=1, label="Minimum run length (frames)")
                    minc = gr.Slider(0.0, 0.95, value=0.0, step=0.05, label="Minimum confidence")
                    keep = gr.Checkbox(value=False, label="Keep <sil>")
                    tim = gr.Checkbox(value=True, label="Show per-unit timings")
                btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                out_seq = gr.Textbox(label="Grapheme units", lines=4, show_copy_button=True)
                out_meta = gr.Markdown()
                out_tbl = gr.Dataframe(
                    headers=["unit", "start (s)", "end (s)", "confidence"],
                    label="Timings",
                    wrap=True,
                )
        btn.click(
            transcribe,
            inputs=[audio, smooth, minf, minc, keep, tim],
            outputs=[out_seq, out_meta, out_tbl],
        )
        audio.stop_recording(
            transcribe,
            inputs=[audio, smooth, minf, minc, keep, tim],
            outputs=[out_seq, out_meta, out_tbl],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0
