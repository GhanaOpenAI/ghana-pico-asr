"""Subcommand dispatcher.

    pico transcribe audio.wav -c model/
    pico transcribe recordings/ -f jsonl -o out.jsonl --timings
    pico hf-dataset ghananlpcommunity/ghana-speech-eval --config waxal_Asante_Twi \
         --split eval -n 100 --score
    pico web -c model/
    pico finetune -c model/best.pt --data-root /data --dry-run
    pico text-to-units corpus.txt -o pairs.jsonl
    pico finetune-data new_domain.txt -o mixed.jsonl
    pico info model/best.pt --vocab
"""

from __future__ import annotations

import argparse
import sys

from .. import config as C
from ..languages import LANGUAGES

_COMMANDS = {
    "transcribe": ("transcribe local audio files or directories", "transcribe"),
    "hf-dataset": ("transcribe a Hugging Face dataset by name", "hf_dataset"),
    "web": ("launch the browser UI", "webui"),
    "finetune": ("fine-tune a checkpoint on a prepared feature store", "finetune"),
    "text-to-units": ("convert text to grapheme units (the reverse direction)", "textunits"),
    "finetune-data": ("build new-domain pairs from raw text, mixed with real replay", "finetune_data"),
    "info": ("show what a checkpoint is and how it was made", "info"),
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pico",
        description=(
            f"{C.PROJECT} — a compact grapheme-unit speech recogniser.\n"
            f"Supported languages: {', '.join(sorted(LANGUAGES))}.\n\n"
            "The model emits a sequence of grapheme units, not words; a separate "
            "text-recovery model turns those into sentences."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    for name, (help_text, module) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(_module=module)
        # Argument definitions live with their command, imported lazily so
        # `pico --help` does not pay for torch.
        p.set_defaults(_needs_args=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0
    if argv[0] == "--version":
        from .. import __version__

        print(f"{C.PROJECT} {__version__}")
        return 0

    command = argv[0]
    if command not in _COMMANDS:
        print(f"unknown command {command!r}\n", file=sys.stderr)
        build_parser().print_help()
        return 2

    _, module_name = _COMMANDS[command]
    import importlib

    module = importlib.import_module(f".{module_name}", __package__)

    ap = argparse.ArgumentParser(prog=f"pico {command}", description=module.__doc__)
    module.add_args(ap)
    args = ap.parse_args(argv[1:])
    return module.run(args)
