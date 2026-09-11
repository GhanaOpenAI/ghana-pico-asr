"""Log-mel feature extraction, shared by dataset preparation and inference."""

from __future__ import annotations

import io

import numpy as np
import torch

from . import config as C


class LogMel:
    """Log-mel spectrogram at 100 frames/sec, kept on the GPU when available."""

    def __init__(self, device: str = "cpu"):
        import torchaudio

        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=C.SAMPLE_RATE,
            n_fft=C.N_FFT,
            win_length=C.WIN_LENGTH,
            hop_length=C.HOP_LENGTH,
            f_min=C.F_MIN,
            f_max=C.F_MAX,
            n_mels=C.N_MELS,
            power=2.0,
            center=True,
        ).to(device)

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """``waveform`` [n_samples] float32 @16 kHz -> log-mel [n_frames, n_mels]."""
        spec = self.mel(waveform.to(self.device).float())
        spec = torch.log10(spec.clamp_min(C.LOG_MEL_FLOOR))
        return spec.transpose(0, 1).contiguous()  # [T, n_mels]


class AudioDecodeError(RuntimeError):
    """A cell that could not be decoded. Callers skip the utterance."""


def _decode_ffmpeg(raw: bytes, target_sr: int) -> np.ndarray:
    """Decode via an ffmpeg subprocess, resampling and downmixing in one pass.

    Deliberately a subprocess. In-process libsndfile decoding of a malformed
    file can corrupt the heap or segfault, and neither is catchable in Python —
    two such crashes each killed a whole 3-hour alignment shard. A subprocess
    crash surfaces as a non-zero exit code, which *is* catchable, so one bad
    file costs one utterance instead of the shard.
    """
    from subprocess import PIPE, Popen

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "1",
        "-i", "pipe:0",
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(target_sr),
        "pipe:1",
    ]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate(raw)
    if proc.returncode != 0 or not out:
        raise AudioDecodeError(
            f"ffmpeg exit {proc.returncode}: {err.decode('utf-8', 'replace')[:200]}"
        )
    wav = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(wav, dtype=np.float32)


def _decode_soundfile(raw: bytes, target_sr: int) -> np.ndarray:
    """In-process decode. Faster, but a malformed file can take the process
    down — see :func:`_decode_ffmpeg`."""
    import soundfile as sf

    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        import torchaudio.functional as AF

        wav = AF.resample(torch.from_numpy(wav), sr, target_sr).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def decode_audio(
    audio_field: dict, target_sr: int = C.SAMPLE_RATE, backend: str = "soundfile"
) -> np.ndarray:
    """Decode a HF ``Audio`` parquet cell to mono float32 at ``target_sr``.

    The parquet column is a struct of ``{bytes, path}``. Corpora arrive at
    different sample rates (24 kHz TTS, 16 kHz elsewhere), so resampling is
    handled by the decoder.

    ``backend`` defaults to ``"soundfile"``. Do not use ``"ffmpeg"`` alongside
    an initialised CUDA context: ``Popen`` forks, and forking a CUDA process
    corrupts the parent heap.
    """
    raw = audio_field.get("bytes")
    if raw is None:
        path = audio_field.get("path")
        if not path:
            raise AudioDecodeError("audio cell has neither bytes nor path")
        with open(path, "rb") as fh:
            raw = fh.read()

    if backend == "soundfile":
        return _decode_soundfile(raw, target_sr)
    if backend == "ffmpeg":
        return _decode_ffmpeg(raw, target_sr)
    raise ValueError(f"unknown decode backend {backend!r}")


def n_frames_for(n_samples: int) -> int:
    """Frame count torchaudio produces for ``n_samples`` with ``center=True``."""
    return n_samples // C.HOP_LENGTH + 1
