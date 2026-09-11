"""Central configuration for the Twi phoneme pipeline.

Every stage (alignment, feature extraction, windowing, training) reads its
constants from here so the dataset on the Modal volume and the model that
consumes it can never drift apart.
"""

from dataclasses import asdict, dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# Audio / acoustic features
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 16_000  # the MMS aligner is hard-wired to 16 kHz
N_FFT = 400  # 25 ms analysis window
WIN_LENGTH = 400
HOP_LENGTH = 160  # 10 ms hop -> 100 frames/sec
N_MELS = 40  # a 300 ms patch is therefore 40 x 30
F_MIN = 20.0
F_MAX = 7600.0
LOG_MEL_FLOOR = 1e-10

FRAME_MS = HOP_LENGTH * 1000 // SAMPLE_RATE  # 10 ms per mel frame

# --------------------------------------------------------------------------- #
# Chunking and temporal context
# --------------------------------------------------------------------------- #
# The model labels EVERY 10 ms frame, not one frame per window. The forced
# aligner already says which unit occupies each frame, so per-frame targets use
# all of that information instead of throwing 4 in 5 units away.
#
# That splits what used to be one number ("window length") into two:
#
#   RECEPTIVE FIELD - how much audio each frame's prediction sees. This is the
#     real co-articulation knob, and what separates ky/gy, tw/dw, hy/hw.
#     Measured median unit duration is 60 ms, so 300 ms of context is ~5 units.
#
#   CHUNK - how much audio is in one training sample. Purely an efficiency
#     choice now, since every frame in the chunk is a supervised label. Longer
#     chunks waste proportionally fewer frames to truncated edge context.

CHUNK_MS = 1000  # 100 frames per training sample -> 100 labels
CHUNK_STRIDE_MS = 500  # 50% overlap between consecutive chunks

#: Dilations of the temporal conv stack — this is the receptive-field knob:
#:     (1, 2, 4)        -> 270 ms   ~4.5 units   (default, matches the 300 ms target)
#:     (1, 2, 4, 8)     -> 430 ms   ~7 units
#:     (1, 2, 4, 8, 16) -> 750 ms   ~12 units
#: Exactly: RF_frames = 1 + 2 * n_freq_convs + 2 * sum(dilations).
TEMPORAL_DILATIONS = (1, 2, 4)

#: Label id for frames that must not contribute to the loss (a unit the
#: aligner was unsure about, or one dropped from the vocabulary).
IGNORE_INDEX = -100

# --------------------------------------------------------------------------- #
# Alignment model
# --------------------------------------------------------------------------- #

ALIGNER_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
ALIGNER_STRIDE_MS = 20.0  # 320 samples @ 16 kHz -> emission frame rate

# --------------------------------------------------------------------------- #
# Source datasets
# --------------------------------------------------------------------------- #

# The full corpus, not the 10k subset it started from. Human transcripts, and
# the best-aligning source we have (-0.70 mean score).
TTS_REPO = "ghanaopenai/new-twi-tts-aligned"
TTS_TEXT_COL = "text"
TTS_N_SHARDS = 54
TTS_TARGET_UTTS = 300_000  # above the 145,258 available, so all of it is used

ASR_REPO = "ghananlpcommunity/twi-health-asr-gemini-500hrs"
ASR_TEXT_COL = "transcription"
ASR_N_SHARDS = 114
# Scaled up after the 10k-utterance models proved data-limited: `wide-ctx`
# began overfitting at epoch 11 of 14, so more data beats a longer schedule.
# Every shard is used now, for maximum speaker/topic spread.
ASR_TARGET_UTTS = 40_000
ASR_N_SHARDS_USED = 114  # all of them

# Kumawood: speech segments from Ghanaian films, ~250 h. `twi_text` is Google
# STT (`ak`) machine output, so transcript quality is a risk — but segments
# average ~5 s rather than a flat 30 s, and short utterances align markedly
# better here (less room for the aligner's free <star> to absorb audio).
KUMA_REPO = "ghananlpcommunity/kumawood-speech-transcriptions"
KUMA_TEXT_COL = "twi_text"
KUMA_N_SHARDS = 100
KUMA_TARGET_UTTS = 180_000  # above the ~173.4k available, so all of it is used
KUMA_N_SHARDS_USED = 100  # all of them, for film/speaker spread

# Additional corpora, added after the 565 h run showed the model was still
# data-limited (no overfitting at 30 labelled frames per parameter).
# A target above the available row count means "use all of it".
EXTRA_SOURCES: dict[str, dict] = {
    "female": {
        "repo": "ghananlpcommunity/ghana-female-twi-asr-16word-splits",
        "n_shards": 14,
        "text_col": "text",
        "transcript": "human",
        "rows": 25_951,
        "note": "female speakers, short word-split utterances",
    },
    "agric": {
        "repo": "ghanaopenai/twi-agriculture-speech",
        "n_shards": 30,
        "text_col": "transcription",
        "transcript": "human",
        "rows": 15_174,
        "note": "agriculture domain, 16 kHz",
    },
    # Not aligned: every shard crashes libsndfile during decoding. Kept
    # registered so id 6 is never reused, but excluded from the default splits.
    "multispk": {
        "repo": "ghanaopenai/twi-speech-text-multispeaker-16k",
        "n_shards": 3,
        "text_col": "text",
        "transcript": "human",
        "rows": 15_560,
        "note": "multi-speaker read speech, 16 kHz — UNALIGNED, decoder crashes",
        "aligned": False,
    },
    "bible": {
        "repo": "ghanaopenai/asante-twi-bible-speech-text",
        "n_shards": 42,
        "text_col": "text",
        "transcript": "human",
        "rows": 34_164,
        "note": "Bible readings, multiple speakers, human transcripts",
    },
}

#: Splits kept out of validation and test. Currently empty.
#:
#: For a corpus that re-transcribes another's audio: without this, the same
#: audio could be trained on under one transcript and evaluated under the
#: other. Set `"train_only": True` on an EXTRA_SOURCES entry.
TRAIN_ONLY_SPLITS: frozenset[str] = frozenset(
    {k for k, v in EXTRA_SOURCES.items() if v.get("train_only")}
)

# Held-out evaluation set: a different corpus entirely, with **human**
# transcripts rather than Gemini's, so it measures generalisation rather than
# agreement with the training labels' own biases.
EVAL_REPO = "ghananlpcommunity/ghana-speech-eval"
#: Shard 0 is the dev slice, for choosing decode settings; shard 1 is held
#: back so reported numbers are not tuned on themselves.
EVAL_FILE = "waxal_Asante_Twi/eval-00000-of-00002.parquet"
EVAL_FILE_HELDOUT = "waxal_Asante_Twi/eval-00001-of-00002.parquet"
EVAL_TEXT_COL = "text"

# --------------------------------------------------------------------------- #
# Utterance-level sanity filters applied during preparation
# --------------------------------------------------------------------------- #

MIN_DURATION_S = 0.4
MAX_DURATION_S = 31.0
MIN_UNITS = 3
# Twi runs ~10-18 grapheme units/sec. Anything far outside that means the
# transcript and the audio disagree (a real risk with Gemini transcripts).
MIN_UNITS_PER_SEC = 3.0
MAX_UNITS_PER_SEC = 32.0

# --------------------------------------------------------------------------- #
# Volume layout
# --------------------------------------------------------------------------- #

VOLUME_NAME = "twi-phoneme"
VOLUME_MOUNT = "/data"

MEL_DIR = "features/mel"
MANIFEST_DIR = "features/manifest"
VOCAB_PATH = "features/vocab.json"
INDEX_DIR = "features/index"
CKPT_DIR = "checkpoints"

HF_CACHE_DIR = "hf-cache"

SIL_TOKEN = "<sil>"

#: Default language. The model itself is language-agnostic; see
#: ghana_pico_asr.languages for the registry.
LANGUAGE = "twi"

PROJECT = "ghana-pico-asr"

# --------------------------------------------------------------------------- #
# Released artefacts on the Hub
# --------------------------------------------------------------------------- #

#: Released weights per language. The tools fetch from here when no local
#: checkpoint is given, which keeps the GitHub repo free of binaries and makes
#: usage of the published model countable.
HF_MODEL_REPO: dict[str, str] = {"twi": "ghanaopenai/ghana-pico-asr-twi"}

#: Weights filename inside a model repo.
HF_WEIGHTS_FILE = "best.pt"

# --------------------------------------------------------------------------- #
# Source ids for published pair data
# --------------------------------------------------------------------------- #
# Published pairs label their origin with a number rather than a corpus name,
# so the dataset does not hard-code dataset identities and the ids stay stable
# if a corpus is renamed or replaced. The mapping is documented in the README
# and the dataset card.
#
# Ids are append-only: never renumber an existing source, or previously
# published pair data becomes mislabelled.
SOURCE_IDS: dict[str, int] = {
    "tts": 1,       # read speech, studio, human transcripts
    "asr": 2,       # health talk shows, machine transcripts
    "kuma": 3,      # film dialogue, machine transcripts
    "female": 4,    # female speakers, short word-split utterances
    "agric": 5,     # agriculture domain
    "multispk": 6,  # multi-speaker read speech, 16 kHz
    "bible": 7,     # Bible readings, multiple speakers
}
SOURCE_NAMES: dict[int, str] = {v: k for k, v in SOURCE_IDS.items()}


def source_id(split: str) -> int:
    """Numeric id for a feature-store split, for published pair data."""
    if split not in SOURCE_IDS:
        raise KeyError(
            f"no source id for {split!r}; add one to SOURCE_IDS (append-only, "
            f"next free id is {max(SOURCE_IDS.values(), default=0) + 1})"
        )
    return SOURCE_IDS[split]



@dataclass
class ChunkConfig:
    """Everything that changes which frames exist and how they are labelled."""

    chunk_ms: int = CHUNK_MS
    chunk_stride_ms: int = CHUNK_STRIDE_MS

    # Threshold defaults come from the measured score distributions, not
    # intuition. Unit scores are extremely skewed: the median unit aligns at
    # -0.05 (TTS) / -0.24 (ASR), but the 1st percentile is around -8. A few
    # catastrophic units therefore wreck an utterance's *mean* score, which is
    # why filtering must happen per unit and not per utterance:
    #
    #   min_utt_score  -1.2 would drop 34% of the health corpus (~28 h)
    #                  -2.0 drops 2.3% of it, and 1.9% of TTS
    #   min_unit_score -1.0 drops 17% of TTS units, 29% of ASR units, whose
    #                  frames become IGNORE_INDEX and contribute no loss
    #
    #: Drop units the aligner was not confident about (mean CTC log-prob).
    #: Their frames become IGNORE_INDEX rather than a wrong label.
    min_unit_score: float = -1.0
    #: Drop only utterances that failed outright, not merely imperfect ones.
    min_utt_score: float = -2.0
    #: Units seen fewer than this many times across the corpus are dropped.
    min_unit_count: int = 200

    #: Frames covered by no unit (pauses, breaths, leading/trailing silence).
    #: Set True to exclude every such frame from the loss instead of labelling
    #: it <sil>.
    silence_as_ignore: bool = False

    #: A gap frame is only called <sil> if its energy sits below this
    #: percentile of the *same utterance's* speech-frame energies.
    #:
    #: Length alone cannot tell a pause from speech the transcript omitted,
    #: and with Gemini-generated transcripts a dropped sentence is a very
    #: likely cause of a long gap — which would otherwise label seconds of
    #: real speech as silence. Energy separates the two: silence is quiet,
    #: untranscribed speech is not. Set to 100 to disable the check.
    silence_energy_percentile: float = 10.0

    #: Minimum gap length to call <sil>. Shorter gaps are excluded from the
    #: loss instead.
    #:
    #: This matters more than it looks. The aligner's <star> wildcard is
    #: appended with log-prob 0 (probability 1), so it is *free* for the
    #: Viterbi path to occupy: wherever a unit scores weakly, frames get
    #: parked in the star at a word boundary rather than in the unit. That
    #: compresses unit spans and inflates inter-word gaps, so a short gap is
    #: usually speech the star swallowed, not a pause. Labelling it <sil>
    #: would teach the model that speech is silence.
    min_silence_gap_ms: int = 120

    #: The health corpus yields ~7x the frames of the TTS corpus. 0 = no cap.
    max_chunks_per_split: int = 0
    #: Sources with usable aligned data on the volume.
    splits: tuple = ("tts", "asr", "kuma", "female", "agric", "bible")

    def key(self) -> str:
        import hashlib
        import json

        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @property
    def chunk_frames(self) -> int:
        return self.chunk_ms // FRAME_MS

    @property
    def stride_frames(self) -> int:
        return max(1, self.chunk_stride_ms // FRAME_MS)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64  # chunks, each carrying chunk_frames labels
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_frac: float = 0.03
    label_smoothing: float = 0.05
    dropout: float = 0.1
    #: Frequency-axis conv stack; each stage halves the mel axis.
    channels: tuple = (32, 64, 128)
    #: Width of the temporal (dilated) trunk.
    temporal_dim: int = 192
    dilations: tuple = TEMPORAL_DILATIONS
    # Inverse-frequency class weighting, tempered: w = (1/freq) ** alpha
    class_weight_alpha: float = 0.3
    # SpecAugment on the 40 x chunk_frames patch
    freq_mask: int = 6
    time_mask: int = 10
    n_masks: int = 2

    num_workers: int = 8
    val_frac: float = 0.02
    test_frac: float = 0.02
    seed: int = 1234
    max_train_chunks: int = 0  # 0 = use all
    #: Utterances from the held-out dev shard to score each epoch. 0 disables.
    #:
    #: Validation runs against the *aligner's* labels, so every metric built on
    #: it inherits that reference's noise floor. Measured over the 30-epoch run,
    #: validation balanced accuracy correlated with validation UER at only
    #: Spearman -0.51, and the 30-epoch run's large validation gains over the
    #: 8-epoch run (0.3211 -> 0.3079) were worth 0.0014 on held-out human
    #: transcripts -- almost all of it was fitting the aligner more closely.
    #:
    #: This scores whole utterances from `EVAL_FILE` (shard 0) against human
    #: transcripts instead, exposing `dev_uer` for `select_metric`. Shard 1
    #: (`EVAL_FILE_HELDOUT`) stays untouched for final reporting.
    dev_utts: int = 0

    #: Validation metric that selects the best checkpoint.
    #:
    #: "unit_error_rate" is the task metric, but it is measured against the
    #: *aligner's* labels, so it bottoms out at that reference's own noise
    #: floor — on this corpus it flattens by ~epoch 3 while per-class
    #: discrimination keeps improving. Selecting on a flat, noisy metric
    #: picks near-arbitrarily between similar models, so balanced accuracy
    #: is the default.
    select_metric: str = "balanced_acc"  # or "unit_error_rate" / "macro_f1" / "dev_uer"

    #: Stop after this many epochs without a `min_delta` improvement in
    #: `select_metric`. 0 disables early stopping.
    #:
    #: Note this sits slightly awkwardly with the cosine schedule, which is
    #: sized to the *full* epoch budget: stopping at 20 of 30 leaves the LR at
    #: ~25% of peak rather than annealed to zero, so the model is not at a
    #: converged point. Best-checkpoint selection protects the result, but if
    #: early stopping fires well before the horizon, a re-run with the shorter
    #: budget (properly annealed) will usually beat the stopped run.
    patience: int = 4
    min_delta: float = 5e-4

    #: Archive every epoch to `checkpoints/<run>/epochs/epoch_NN.pt`, not just
    #: the selected one.
    #:
    #: `best.pt` is overwritten by the next improvement, so a run can only ever
    #: be reconsidered at the epoch its selection metric happened to prefer. On
    #: the 30-epoch run the best-UER epoch (16) was gone by epoch 20, leaving no
    #: way to score it on the held-out set -- and the selection metric and the
    #: task metric disagreed by 0.012 UER. An epoch archive is ~46 MB; a run is
    #: ~$21. Keeping them is not the expensive side of that trade.
    keep_epoch_checkpoints: bool = True
    log_every: int = 100
    run_name: str = "twi2dcnn"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
