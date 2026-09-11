# ghana-pico-asr

A compact speech recogniser for Ghanaian languages that outputs **grapheme
units** rather than words. Currently supports **Twi (Asante)**; the model,
training and inference code are language-agnostic and adding a language means
adding one module.

```
audio ──▶ 2D CNN ──▶ ɔ y ɛ  n e  h o  a dw u m a
                     one unit per 10 ms frame, runs collapsed
```

It is deliberately **not** a full ASR system. It recovers *what sounds were
produced*, not what was said. A separate text-recovery model turns unit
sequences into sentences — see [Roadmap](#roadmap).

---

## Why grapheme units

Akan orthography spells several single phonemes with two letters, so the unit
inventory is the single letters plus those digraphs:

```
"nkyekyɛmu"  ->  n | ky | e | ky | ɛ | m | u
"ahwɛ"       ->  a | hw | ɛ
"adwuma"     ->  a | dw | u | m | a
```

For Twi that is `ky gy hy ny tw dw kw gw hw nw` plus single letters — 37
classes including `<sil>`. Nasal + stop clusters (`nk`, `nt`, `mp`, …) are
deliberately *not* digraphs: they are two phonemes.

The payoff shows in the results — those digraphs are among the model's **best**
classes despite tiny support, because a 2D convolution reads formant structure
across frequency and its movement across time at once.

---

## Install

```bash
pip install git+https://github.com/ghanaopenai/ghana-pico-asr.git
```

or, from a clone:

```bash
pip install -e .                  # inference
pip install -e '.[datasets]'      # + read HF datasets directly
pip install -e '.[web]'           # + browser UI
pip install -e '.[all]'           # + training and Modal
```

## Use

**No download step.** With no checkpoint given, the released weights are pulled
from [`ghanaopenai/ghana-pico-asr-twi`](https://huggingface.co/ghanaopenai/ghana-pico-asr-twi)
and cached. To use your own, pass `-c/--checkpoint` (a file, a directory
containing `best.pt`, or another HF repo id), or set `$PICO_CHECKPOINT`.

```bash
# one file, or a directory searched recursively
pico transcribe utterance.wav
pico transcribe recordings/ -f jsonl -o out.jsonl --timings

# any Hugging Face dataset, by name — the audio column is detected and validated
pico hf-dataset ghananlpcommunity/ghana-speech-eval \
     --config waxal_Asante_Twi --split eval -n 100 --score

# browser UI: record or upload, see units and per-unit timings
pico web

# text -> units (deterministic; see Roadmap)
pico text-to-units corpus.txt -o pairs.jsonl
echo "Ɔyɛ ne ho adwuma" | pico text-to-units -

# training data for the text-recovery stage — one command, no audio needed
pico finetune-data new_domain.txt -o mixed.jsonl

# what is this checkpoint, and what was it trained on?
pico info model/best.pt --vocab

# fine-tune on your own prepared data
pico finetune -c model/best.pt --data-root /data --dry-run
```

`hf-dataset` refuses politely when a dataset has no audio:

```
this dataset has no audio column.
  columns found: ['text', 'label']
  Pass --audio-column NAME if one of these holds audio, or pick a
  dataset with an Audio feature.
```

### Python

```python
from ghana_pico_asr.infer import UnitTagger

tagger = UnitTagger("model/best.pt")
tagger.transcribe("utterance.wav")        # 'ɔ y ɛ n e h o'
for u in tagger.units("utterance.wav"):
    print(u.unit, round(u.start, 3), round(u.end, 3), round(u.confidence, 3))
```

Whole file, one forward pass, no segmentation and no text needed. The
checkpoint carries its own vocabulary, normalisation statistics, feature
configuration and architecture, so nothing is re-specified at load time.

---

## Results

Evaluated on [`ghananlpcommunity/ghana-speech-eval`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval)
`waxal_Asante_Twi` — a corpus with **human** transcripts, unseen in training.
Shard 0 chooses decode settings; shard 1 is never tuned on, so it is the honest
number. Unit error rate = edit distance after collapsing repeats.

| model | params | context | train audio | epochs | **UER** | len ratio |
|---|---|---|---|---|---|---|
| **`hf30`** | 12.0 M | 750 ms | **805 h** | 30 | **0.3058** | 0.928 |
| `xl` | 12.0 M | 750 ms | 805 h | 8 | 0.3079 | 0.924 |
| `final` | 3.99 M | 750 ms | 565 h | 14 | 0.3332 | 0.900 |
| `wide-ctx` | 3.99 M | 750 ms | 93.8 h | 14 | 0.3590 | 0.888 |
| `big-ctx` | 3.39 M | 430 ms | 93.8 h | 14 | 0.3755 | 0.885 |
| `big-cap` | 2.80 M | 270 ms | 93.8 h | 14 | 0.4000 | 0.890 |
| baseline | 862 k | 270 ms | 93.8 h | 14 | 0.4146 | 0.880 |

**0.4146 → 0.3058 is a 26% relative improvement.** What moved it, in
descending order:

1. **Context, 270 → 750 ms.** The largest single win. `big-cap` isolates it by
   adding capacity at fixed context, so capacity alone gives 0.4146 → 0.4000
   and widening context on top gives 0.4000 → 0.3590. Dilated convolutions
   make this cheap: 750 ms costs 1.2 M more parameters than 270 ms and 13% more
   time per epoch.
2. **Capacity, 4.0 → 12.0 M**, with data at 805 h: 0.3332 → 0.3079, 7.6%.
3. **Data, 93.8 → 565 h.** 7.5% relative, measured cleanly — `final` and
   `wide-ctx` differ only in corpus.
4. **Decode tuning.** `smooth=7` over `smooth=9` is worth ~0.003 for free, and
   has to be re-swept after a capacity change.

**Epochs are not on this list.** `xl` and `hf30` differ only in schedule length
— 8 epochs versus 30, nearly 4× the compute — and differ by 0.002 UER, which is
inside the noise of a 400-utterance sample. The 30-epoch run improved much more
on *validation* (0.3211 → 0.3079), but that gain was fitting the aligner's
labels rather than transcribing better. This model is near the ceiling its
label quality supports; better alignment is the remaining lever.

Per-class recall (baseline model) — no class collapsed despite a 1,942×
imbalance, and the digraphs that motivate the whole approach do well:

```
ny 0.826   hw 0.799   dw 0.742   gy 0.673   hy 0.623   ky 0.610   tw 0.590
```

`tw`/`dw` cross-confusion is essentially zero (0.000/0.002).

### On the ɛ/e and ɔ/o contrasts

These pairs romanise identically for the aligner, so it cannot label them —
the model can only learn them from audio, or from ATR vowel harmony context.
`modal_app/context_probe.py` separates the two by masking audio:

```
                    270 ms model     750 ms model
whole utterance        0.6756           0.7116
span only (acoustics)  0.5554           0.5371
context reliance       -17.8%           -24.5%
```

Both keep 75–82% of their accuracy with **all** surrounding audio removed, so
decisions are substantially acoustic. But the wider model is more
context-dependent, and on ɛ/e it is *worse* acoustically (isolated 0.398 vs
0.438) despite scoring better overall — its advantage there is harmony
inference, not better vowel acoustics.

**This matters only if you need phonetic fidelity** (pronunciation scoring,
mispronunciation detection, phonetic research), where the narrower model is
preferable. For recognition, context reliance is a feature.

---

## Architecture

```
[B, 1, 40, T]  log-mel, 100 frames/sec
   │  frequency stack: 3 × (2× conv3×3 + BN + ReLU + maxpool on the mel axis only)
   │  full-height conv collapses the mel axis
[B, 384, T]
   │  temporal trunk: residual dilated conv1d, dilations 1,2,4,8,16
[B, n_classes, T]  one distribution per 10 ms frame
```

Nothing downsamples time and nothing is fully connected, so the same weights
run on a 1 s training chunk or a 30 s utterance. Features are 16 kHz, 25 ms
windows, 10 ms hop, 40 mels, log10.

**Receptive field** — how much audio each frame prediction sees — is set by the
dilation stack, not by a window parameter:

| dilations | context | units of context |
|---|---|---|
| `1,2` | 130 ms | ~2 |
| `1,2,4` | 270 ms | ~4.5 |
| `1,2,4,8` | 430 ms | ~7 |
| **`1,2,4,8,16`** | **750 ms** | **~12** |

At the measured 60 ms median unit, 750 ms spans ~6 units either side. Note the
context is *centred*, so streaming would carry ~370 ms of inherent latency;
causal dilations would remove that at some accuracy cost.

---

## Roadmap: the text-recovery stage

The unit sequence is an intermediate representation. The planned second stage
is a **T5 text-recovery model** that turns units into sentences, trained on
this model's real output against reference transcripts.

This repo produces both kinds of training data it needs.

### The bundled real pairs

Real pairs — the recogniser's own output against known transcripts — ship with
the package as `ghana_pico_asr/data/real_pairs_twi.csv`. There is no step to
run: `pico finetune-data` reads them directly, so building new-domain training
data needs no audio, no GPU and no checkpoint.

Each pair records which corpus it came from as a **numeric `source` id**, so
the replay mix can be spread across domains without the dataset hard-coding
corpus identities:

| id | speech | transcripts | share of pairs |
|---|---|---|---|
| **1** | read speech, studio | human | 5% |
| **2** | health talk shows | machine (Gemini) | 20% |
| **3** | film dialogue | machine (Google STT) | 75% |

Ids are append-only — an existing one is never renumbered, or already-published
pairs would be mislabelled. The mapping lives in `config.SOURCE_IDS`.

Their unit side carries the model's actual error profile, which is what the
recovery model must learn to repair:

```
71.0% match   15.2% substituted   13.8% deleted   4.3% inserted
```

Two properties of that profile shape the recovery model:

* **Deletions outnumber insertions 3.2:1**, so it mostly has to *insert*, not
  substitute. Geminates are a minor part of this — only 6% of adjacent-duplicate
  reference units were dropped.
* **Substitutions are almost entirely vowel-for-vowel** (`a→ɛ`, `ɔ→o`, `o→u`,
  `i→e`, `e→ɛ`…). Consonants are largely solid; vowel *quality* is the weak
  point.

Both are errors a text model can fix from context, since Twi vowel harmony and
lexical structure strongly constrain which vowel is legal in a word.

Maintainers regenerate the CSV once per released model with
`scripts/build_real_pairs.py`; users never touch it.

### New-domain data — one command, no audio

```bash
pico finetune-data legal_corpus.txt -o mixed.jsonl
pico finetune-data --hf-dataset some/corpus --text-column body -n 20000 -o mixed.jsonl
```

Converts raw text to clean units and mixes in a replay sample of the bundled
real pairs. The replay matters: synthetic pairs alone are 100% clean, and
training on only those would teach the recovery model that its input is
trustworthy — undoing the repair behaviour. The real pairs hold it in place
while the synthetic ones supply the new vocabulary.

The corpus is heavily unbalanced (source 3 is ~75% of it), so `--balance`
controls how the replay is spread:

```bash
pico finetune-data corpus.txt --balance proportional   # default
pico finetune-data corpus.txt --balance equal          # even across sources
```

`proportional` mirrors each source's real share, which represents the error
profile as it actually occurs. `equal` stops one domain dominating, at the cost
of over-representing the smaller corpora. Either way the report shows the
achieved split:

```
replay by source        1=150  2=600  3=2,250   (proportional)
replay by source        1=500  2=1,068  3=1,432 (equal)
```

`--ratio` is parts real per part synthetic (default 1.0, an even mix). When
fewer real pairs exist than the ratio asks for, **all** of them are used rather
than sampled with replacement, and the report states the ratio actually
achieved:

```
synthetic (new domain)      20,000
real (replay)                5,000  of 5,000 available
total                       25,000
ratio real:synthetic         0.250 (requested 1.0)
note: wanted 20,000 real pairs but only 5,000 exist, so all were used.
```

Real pairs come from the bundled sample by default, or from anywhere:

```bash
pico finetune-data corpus.txt --real my_pairs.csv                        # local
pico finetune-data corpus.txt --real ghananlpcommunity/twi-grapheme-unit-pairs
```

`pico text-to-units` is the deterministic text→units mapping used for the
synthetic side. It calls **the same function that produced this model's
training labels**, so both directions share one inventory by construction — if
they drifted, the recovery model would learn a mapping the recogniser never
emits.

Adding a language means one module in `ghana_pico_asr/languages/` — an
inventory, a fold map, a romanisation map and contrast pairs. The model,
dataset, trainer and CLI need no changes.

---

## Training

See [TRAINING.md](TRAINING.md) for the two-stage pipeline (CTC forced alignment
→ frame labels → training), the data hazards this corpus actually has, and how
to reproduce the results above on Modal.

## Layout

```
ghana_pico_asr/
  languages/     pluggable language definitions (base.py + twi.py)
  config.py      constants and the two config dataclasses
  features.py    log-mel + audio decoding
  align.py       CTC forced alignment of grapheme units
  prepare.py     feature-store builder (pure Python)
  dataset.py     frame-label tracks + chunk serving
  model.py       PicoASRNet
  trainer.py     training loop, evaluation, contrast report
  infer.py       whole-utterance decode
  finetune.py    checkpoint transfer with vocabulary remapping
  provenance.py  what a checkpoint records about its own making
  pairs.py       (units, text) pairs and the replay mix
  data/          bundled real (units, text) pairs
  cli/           transcribe · hf-dataset · web · finetune
                 finetune-data · text-to-units · info
scripts/         maintainer tools (regenerate the bundled pairs)
modal_app/       training, evaluation and probes on Modal
tests/           79 tests, no GPU / network / Modal needed
```

## Licence

Code CC-BY-NC-4.0. The training corpora are CC-BY-NC-4.0, so models trained
from them are **non-commercial**. See [MODEL_CARD.md](MODEL_CARD.md).
