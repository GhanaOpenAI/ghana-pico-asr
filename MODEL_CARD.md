# Model card — ghana-pico-asr (Twi)

## What it is

A 12.0 M-parameter 2D CNN that maps 16 kHz audio to a sequence of **Twi
grapheme units** — one label per 10 ms frame, runs of identical labels
collapsed. 37 classes: the single letters of Akan orthography, the ten digraphs
that spell single phonemes (`ky gy hy ny tw dw kw gw hw nw`), and `<sil>`.

It is **not a speech-to-text system.** It outputs sounds, not words. Turning
`ɔ y ɛ n e h o` into "Ɔyɛ ne ho" requires a separate text-recovery model.

## Intended use

- A front end for a Twi ASR pipeline, paired with a text-recovery model.
- Phoneme-level analysis of Twi speech: inventory frequencies, durations,
  segment timings.
- A starting point for fine-tuning to another Ghanaian language or domain.

## Out-of-scope and cautions

- **Not for phonetic or pronunciation assessment as-is.** The 750 ms model
  gets part of its accuracy from context rather than from the target segment's
  own acoustics; on ɛ/e it is *worse* acoustically than a 270 ms model
  (isolated accuracy 0.398 vs 0.438). It will tend to "correct" what a speaker
  actually said toward what it expects — exactly the errors a pronunciation
  scorer must surface. Use a narrow-context variant for that.
- **Non-commercial only.** These weights are CC-BY-NC-4.0, inherited from the
  training corpora. The *code* that produced them is MIT, so retraining on a
  corpus you may use commercially carries no such restriction.
- **Not a language identifier.** It will happily emit Twi units for audio in
  any language.
- **English code-switching is filtered, not modelled.** See below.

## Performance

Held-out [`ghananlpcommunity/ghana-speech-eval`](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech-eval)
`waxal_Asante_Twi` **shard 1** — human transcripts, an unseen corpus, never
used for tuning. 400 utterances (~53k reference units).

| metric | value |
|---|---|
| **unit error rate** | **0.3058** |
| length ratio (decoded / reference units) | 0.928 |

On the aligner-labelled test split (83,038 chunks), for the same checkpoint:

| metric | value |
|---|---|
| frame accuracy | 0.8457 |
| balanced accuracy | 0.7978 |
| macro F1 | 0.7674 |
| unit error rate | 0.3154 |

Roughly one insertion/deletion/substitution per three reference units. The
0.928 length ratio means it emits ~7% fewer units than the reference — it
declines to guess short, ambiguous segments, and forcing them out measurably
*worsens* UER.

### How it got here, and what actually helped

| model | params | epochs | Waxal UER |
|---|---|---|---|
| `final` | 4.0 M | 14 | 0.3332 |
| `xl` | 12.0 M | 8 | 0.3079 |
| **`hf30`** (released) | 12.0 M | 30 | **0.3058** |

**Capacity was the lever; epochs were not.** Tripling parameters bought 0.025
UER (7.6%). Going from 8 epochs to 30 — nearly four times the compute — bought
0.002, which is inside the noise of a 400-utterance sample. The 30-epoch run's
validation UER improved far more than that (0.3211 to 0.3079), but almost none
of it transferred to human transcripts: the extra epochs were fitting the
*aligner's* labels more closely, not transcribing better.

Taken together with the alignment-quality figures below, this model is close to
the ceiling its training labels support. Further gains are far more likely to
come from better alignment than from more capacity or more epochs.

## Training data

805.0 h across six corpora, of which **482.1 h (59.9%) carries a usable frame
label**.

| corpus | utterances | hours | transcripts | mean align score |
|---|---|---|---|---|
| [`new-twi-tts-aligned`](https://huggingface.co/datasets/ghanaopenai/new-twi-tts-aligned) | 42,488 | 45.5 | human | −0.70 |
| [`twi-health-asr-gemini-500hrs`](https://huggingface.co/datasets/ghananlpcommunity/twi-health-asr-gemini-500hrs) | 42,993 | 358.4 | machine (Gemini) | −1.07 |
| [`kumawood-speech-transcriptions`](https://huggingface.co/datasets/ghananlpcommunity/kumawood-speech-transcriptions) | 158,088 | 214.0 | machine (Google STT) | −1.06 |
| [`ghana-female-twi-asr-16word-splits`](https://huggingface.co/datasets/ghananlpcommunity/ghana-female-twi-asr-16word-splits) | 20,237 | 28.2 | human | −0.93 |
| [`twi-agriculture-speech`](https://huggingface.co/datasets/ghanaopenai/twi-agriculture-speech) | 13,608 | 113.4 | human | −0.69 |
| [`asante-twi-bible-speech-text`](https://huggingface.co/datasets/ghanaopenai/asante-twi-bible-speech-text) | 33,079 | 63.8 | human | −0.30 |

294,005 utterances in total. Alignment scores are log-probabilities: the Bible
readings align best (−0.30) and the machine-transcribed corpora worst (−1.07),
which is the single clearest predictor of label quality.

The derived feature store is published at
[`ghanaopenai/twi-grapheme-unit-features`](https://huggingface.co/datasets/ghanaopenai/twi-grapheme-unit-features),
so the GPU alignment pass does not have to be repeated to reproduce or extend
this model.

**Labels are machine-generated.** Frame labels come from CTC forced alignment
(`MahmoudAshraf/mms-300m-1130-forced-aligner`) against transcripts that are
themselves largely machine output. The model inherits those biases. Reported
validation/test UER is agreement with the aligner, which is why the held-out
human-transcribed number above is the one quoted.

### What was excluded, and why

| | share | hours |
|---|---|---|
| labelled (trained on) | 59.9% | 482.1 |
| — of which `<sil>`, quiet gaps ≥120 ms | 19.8% | 159.1 |
| **excluded — loud gaps** | **22.9%** | **184.4** |
| excluded — short gaps | 9.4% | 75.3 |
| excluded — low-scoring utterances (16,488) | 7.9% | 63.2 |

The 184.4 h of "loud gaps" is speech-level audio the aligner assigned no units
to — dropped transcript, plus music and background. Labelling it `<sil>` would
have taught the model that speech is silence, so a gap is only called silence
when it is both ≥120 ms and quieter than the 10th percentile of that
utterance's own speech energy.

### English code-switching

Ghanaian Twi speech mixes in English, which this inventory does not model: no
English digraphs (`ch sh th ph ck qu`) and the Twi rules misfire (`twelve` →
`tw·e·l·v·e`, treating `tw` as /tɕʷ/). The alignment-score filter handles the
worst of it — letters with no 1:1 phone are mostly discarded (`c` 68.6%, `x`
65.3%), while phonemically transparent ones survive and are learned correctly
(`j` 17.7%, `v` 18.6%, `z` 17.7% dropped, comparable to native Twi at 27.9%).
So English is **excluded rather than mislabelled**, which is benign but means
code-switched segments will be weak.

Native digraphs are dropped hardest of all (`ky` 54.3%, `dw` 53.8%): the MMS
aligner has no representation for /tɕ/ or /dʑʷ/. They still learn well, but
improving *alignment* for digraphs is the largest available data win.

## Evaluation caveats

- 400 utterances still cannot resolve small differences; treat anything under
  ~0.005 UER as noise. The 8-epoch and 30-epoch models differ by less than
  that.
- Waxal is prompted image-description speech — cleaner than the film and
  talk-show data that dominates training, so real-world spontaneous speech will
  likely be worse.
- Decode settings (`smooth_frames=7, min_frames=4`) were selected on shard 0
  and reported on shard 1.

## Provenance

Checkpoints record their corpora, hours, alignment model, label policy, feature
configuration, git commit and training hyperparameters:

```bash
pico info model/best.pt          # human-readable
pico info model/best.pt --json   # the full block
```

Checkpoints published before provenance recording show `?` for those fields but
remain fully usable.

## Training a text-recovery model on top

The recogniser's own output, paired with reference transcripts for all 282,088
training utterances, is published at
[`ghanaopenai/twi-grapheme-unit-pairs`](https://huggingface.co/datasets/ghanaopenai/twi-grapheme-unit-pairs)
— so a second-stage model learns to repair the errors this model actually
makes, not hypothetical ones.

```bash
pico finetune-data --new-domain your_text.txt --out domain_pairs.csv
```

New-domain data needs no audio: text is segmented into units deterministically
and mixed 1:1 with real pairs pulled from the Hub.

## Fine-tuning

```bash
pico finetune -c model/best.pt --data-root /your/feature/store --dry-run
```

Weights transfer by unit *name*, so a different vocabulary is safe: shared
units keep their trained weights, new units start fresh, and a mismatched
feature configuration is refused rather than silently producing a model that
predicts noise. Optimiser state is deliberately not inherited — fine-tuning
wants a fresh schedule.

## Citation

```bibtex
@software{ghana_pico_asr,
  title  = {ghana-pico-asr: a compact grapheme-unit speech recogniser for Ghanaian languages},
  author = {GhanaOpenAI},
  year   = {2026},
  url    = {https://github.com/ghanaopenai/ghana-pico-asr}
}
```
