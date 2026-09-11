"""Inference: label every frame of an utterance, then collapse runs into units.

The model is fully convolutional in time, so a whole utterance goes through in
one pass — there is no window to slide, nothing to stitch, and no segmentation
step. A run of identical frame labels becomes one unit, which is how a single
forward pass yields a variable number of units.

    from ghana_pico_asr.infer import UnitTagger

    tagger = UnitTagger("best.pt")
    tagger.transcribe("utterance.wav")        # 'ɔ y ɛ n e h o'
    for u in tagger.units("utterance.wav"):
        print(u.unit, round(u.start, 3), round(u.end, 3), round(u.confidence, 3))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import config as C
from . import features as F
from .languages import get_language
from .model import PicoASRNet


@dataclass
class DecodedUnit:
    unit: str
    start: float  # seconds
    end: float  # seconds
    confidence: float  # mean posterior over the run of frames
    n_frames: int


def surviving_runs(
    probs: np.ndarray,
    min_frames: int = 4,
    min_confidence: float = 0.0,
    drop_class: int | None = None,
):
    """Collapse identical-argmax runs, yielding ``(cls, start, n, confidence)``.

    Shared by inference and by the trainer's dev-set scoring so the two cannot
    drift: a selection metric computed with different decode rules than the
    released decoder would select the wrong checkpoint.
    """
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    i = 0
    while i < len(pred):
        j = i
        while j + 1 < len(pred) and pred[j + 1] == pred[i]:
            j += 1
        n = j - i + 1
        cls = int(pred[i])
        run_conf = float(conf[i : j + 1].mean())
        if n >= min_frames and run_conf >= min_confidence and cls != drop_class:
            yield cls, i, n, run_conf
        i = j + 1


class UnitTagger:
    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        median_frames: list[float] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)

        self.vocab: list[str] = ckpt["vocab"]
        self.mean: float = ckpt["norm"]["mean"]
        self.std: float = ckpt["norm"]["std"]
        tc = ckpt["train_config"]

        self.model = PicoASRNet(
            n_classes=ckpt["n_classes"],
            channels=tuple(tc["channels"]),
            temporal_dim=tc["temporal_dim"],
            dilations=tuple(tc["dilations"]),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.receptive_field_ms = ckpt.get("receptive_field_ms")
        self.language = get_language(ckpt.get("language"))
        self.provenance = ckpt.get("provenance", {})
        # Median span length per class, for recovering geminates and long
        # vowels that repeat-collapsing would otherwise merge. Checkpoints
        # written before this existed carry none, so it can be supplied.
        self.median_frames = median_frames or ckpt.get("unit_median_frames")
        if self.median_frames is not None and len(self.median_frames) != len(self.vocab):
            raise ValueError(
                f"median_frames has {len(self.median_frames)} entries, "
                f"expected {len(self.vocab)} (one per class)"
            )
        self.logmel = F.LogMel(device=self.device)

    # ------------------------------------------------------------------ #

    def _load(self, audio) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        import soundfile as sf

        wav, sr = sf.read(audio, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != C.SAMPLE_RATE:
            import torchaudio.functional as AF

            wav = AF.resample(torch.from_numpy(wav), sr, C.SAMPLE_RATE).numpy()
        return np.ascontiguousarray(wav, dtype=np.float32)

    @staticmethod
    def smooth(probs: np.ndarray, width: int) -> np.ndarray:
        """Moving average over ``width`` frames along time.

        Frame predictions flicker between neighbouring labels, and under
        repeat-collapsing every flicker becomes a spurious unit — the model
        emitted 95 units for a 63-unit reference before smoothing. Averaging
        the posteriors over a few frames removes the flicker without moving
        real boundaries, since a real unit spans ~6 frames.
        """
        if width <= 1:
            return probs
        pad = width // 2
        padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(width, dtype=np.float32) / width
        out = np.empty_like(probs)
        for c in range(probs.shape[1]):
            out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")[: probs.shape[0]]
        return out

    @torch.inference_mode()
    def posteriors(self, audio, max_frames: int = 6000) -> np.ndarray:
        """``[n_frames, n_classes]`` — one distribution per 10 ms frame.

        Long audio is processed in overlapping tiles whose margins are
        discarded, so the result is identical to a single pass but bounded in
        memory. The margin is the receptive field, which is exactly the context
        an interior frame needs.
        """
        mel = self.logmel(torch.from_numpy(self._load(audio)))  # [T, n_mels]
        T = mel.shape[0]
        x = ((mel - self.mean) / self.std).T.unsqueeze(0).unsqueeze(0)  # [1,1,n_mels,T]

        if T <= max_frames:
            logits = self.model(x.to(self.device))
            return torch.softmax(logits.float(), 1)[0].T.cpu().numpy()

        margin = self.model.receptive_field()
        out = np.empty((T, len(self.vocab)), dtype=np.float32)
        pos = 0
        while pos < T:
            lo = max(0, pos - margin)
            hi = min(T, pos + max_frames + margin)
            logits = self.model(x[:, :, :, lo:hi].to(self.device))
            probs = torch.softmax(logits.float(), 1)[0].T.cpu().numpy()
            keep_hi = min(pos + max_frames, T)
            out[pos:keep_hi] = probs[pos - lo : keep_hi - lo]
            pos = keep_hi
        return out

    # ------------------------------------------------------------------ #

    def units_from_posteriors(
        self,
        probs: np.ndarray,
        min_confidence: float = 0.0,
        min_frames: int = 4,
        drop_silence: bool = True,
        split_long_runs: bool = False,
        smooth_frames: int = 7,
    ) -> list[DecodedUnit]:
        """Collapse runs of identical argmax labels into units.

        ``smooth_frames`` averages the posteriors over time before collapsing,
        and ``min_frames`` then drops any run shorter than that many frames.
        Both exist to stop prediction flicker becoming spurious units; the
        measured median unit is 6 frames, so a 4-frame floor discards runs too
        short to be real.

        ``split_long_runs`` tries to recover doubled units: collapsing alone
        turns "hyɛɛ" into "hyɛ" and "nnipa" into "nipa", since two identical
        adjacent units form one continuous run of identical labels. A run much
        longer than that class's median span is re-split.

        Defaults to **off**: it degrades UER on real audio, where an uncertain
        model emits smeared runs that split into spurious repeats. Requires a
        checkpoint carrying ``unit_median_frames``; inert without one.
        """
        probs = self.smooth(probs, smooth_frames)
        sec = C.FRAME_MS / 1000.0

        out: list[DecodedUnit] = []
        for cls, i, n, run_conf in surviving_runs(
            probs,
            min_frames=min_frames,
            min_confidence=min_confidence,
            drop_class=self.vocab.index(C.SIL_TOKEN) if drop_silence else None,
        ):
            unit = self.vocab[cls]
            reps = 1
            if split_long_runs and self.median_frames and unit != C.SIL_TOKEN:
                med = self.median_frames[cls]
                if med and med > 0:
                    # round() rather than floor: a run has to be closer to
                    # 2x the median than to 1x before it becomes two units.
                    reps = max(1, min(3, int(round(n / med))))
            step = n / reps
            for r in range(reps):
                out.append(
                    DecodedUnit(
                        unit=unit,
                        start=(i + r * step) * sec,
                        end=(i + (r + 1) * step) * sec,
                        confidence=run_conf,
                        n_frames=int(round(step)),
                    )
                )
        return out

    def units(self, audio, **kw) -> list[DecodedUnit]:
        """Decode ``audio`` straight to units.

        Note the class stores its label inventory as ``self.vocab``, not
        ``self.units`` — an attribute of that name would shadow this method.
        """
        return self.units_from_posteriors(self.posteriors(audio), **kw)

    def transcribe(self, audio, **kw) -> str:
        """Space-separated grapheme units, e.g. ``'ɔ y ɛ n e h o'``."""
        return " ".join(u.unit for u in self.units(audio, **kw))
