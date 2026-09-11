"""Shared Modal app, image and volume for the Twi phoneme pipeline.

Nothing here is tied to one workspace: the volume is created on demand and the
app name is just a name, so selecting a profile is all it takes to run the same
pipeline elsewhere. Only the HF token secret is named differently per
workspace, hence ``PICO_HF_SECRET``::

    export MODAL_PROFILE=ghana-nlp                        # secret "huggingface"
    export MODAL_PROFILE=michseth PICO_HF_SECRET=huggingface-secret
"""

import os

import modal

from ghana_pico_asr import config as C

APP_NAME = "twi-phoneme-2dcnn"

volume = modal.Volume.from_name(C.VOLUME_NAME, create_if_missing=True)
# Resolved when the app is built, so it must be set for `modal deploy`/`modal
# run`, not for the spawned call.
HF_SECRET_NAME = os.environ.get("PICO_HF_SECRET", "huggingface")
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

TORCH = ["torch==2.5.1", "torchaudio==2.5.1"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "build-essential", "libsndfile1")
    # torch first, in its own layer: the aligner's requirements would otherwise
    # pull whatever torch build pip feels like.
    .pip_install(*TORCH, index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "numpy<2",
        "pyarrow>=15",
        "soundfile>=0.12",
        "huggingface_hub>=0.25",
        "transformers>=4.44,<5",
        "hf_transfer>=0.1.6",
    )
    # pybind11-only build, so default build isolation is fine.
    .pip_install("git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": f"{C.VOLUME_MOUNT}/{C.HF_CACHE_DIR}",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("ghana_pico_asr")
)

app = modal.App(APP_NAME, image=image)

VOLUMES = {C.VOLUME_MOUNT: volume}
