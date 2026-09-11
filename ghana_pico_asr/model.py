"""The 2D CNN grapheme-unit classifier — fully convolutional, one label per frame.

Input  ``[B, 1, N_MELS, T]``  log-mel, 100 frames/sec
Output ``[B, n_classes, T]``  a distribution per 10 ms frame

Two stages, doing two different jobs:

1. **Frequency stack** — 2D convs that halve the mel axis each stage while
   leaving time untouched. This is where the "2D" earns its keep: 3x3 kernels
   see formant structure across frequency *and* its movement across time, which
   is what the Twi palatal/labialised contrasts (ky/gy, tw/dw, hy/hw) and the
   ɛ/e, ɔ/o vowel pairs actually live in. The mel axis is then collapsed by a
   full-height convolution, giving one feature vector per frame.

2. **Temporal trunk** — residual 1D convs with growing dilation. This is what
   sets the receptive field, i.e. how much context each frame's prediction
   sees, without any pooling that would blur frame timing. Dilations
   (1, 2, 4, 8) reach ~300 ms, about five units at the measured 60 ms median.

Because nothing downsamples time and nothing is fully connected, the same
weights run on a 1 s training chunk or a 30 s utterance unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class FreqBlock(nn.Module):
    """Two 3x3 convs then halve the frequency axis only."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1), ceil_mode=True),  # frequency only
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DilatedBlock(nn.Module):
    """Residual dilated temporal conv; preserves length exactly."""

    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class PicoASRNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_mels: int = C.N_MELS,
        channels: tuple = (32, 64, 128),
        temporal_dim: int = 192,
        dilations: tuple = C.TEMPORAL_DILATIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.dilations = tuple(dilations)
        self.n_freq_convs = 2 * len(channels)

        blocks = []
        c_in = 1
        freq = n_mels
        for c_out in channels:
            blocks.append(FreqBlock(c_in, c_out))
            c_in = c_out
            freq = -(-freq // 2)  # ceil, matching ceil_mode=True
        self.freq_stack = nn.Sequential(*blocks)

        # Collapse whatever is left of the mel axis into the channel dim.
        self.collapse = nn.Sequential(
            nn.Conv2d(c_in, temporal_dim, (freq, 1), bias=False),
            nn.BatchNorm2d(temporal_dim),
            nn.ReLU(inplace=True),
        )

        self.temporal = nn.Sequential(
            *[DilatedBlock(temporal_dim, d, dropout) for d in self.dilations]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(temporal_dim, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, n_mels, T]`` -> ``[B, n_classes, T]``."""
        h = self.freq_stack(x)
        h = self.collapse(h).squeeze(2)  # [B, temporal_dim, T]
        h = self.temporal(h)
        return self.head(h)

    # ------------------------------------------------------------------ #

    def receptive_field(self) -> int:
        """Context each output frame sees, in mel frames.

        Every 3x3 frequency conv adds 1 frame either side; every dilated
        temporal conv adds ``dilation`` frames either side.
        """
        return 1 + 2 * self.n_freq_convs + 2 * sum(self.dilations)

    def receptive_field_ms(self) -> int:
        return self.receptive_field() * C.FRAME_MS

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
