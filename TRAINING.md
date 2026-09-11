# Training

Two stages on [Modal](https://modal.com). Stage 1 aligns audio to grapheme
units and stores **timings, not training samples** — so labelling policy,
chunking and balance are all stage-2 decisions that re-run in minutes.

```
stage 1  modal_app/prepare.py   A10G x20   parquet shard -> aligned feature store
stage 2  modal_app/train.py     A10G       feature store -> model
```

```bash
export MODAL_PROFILE=ghana-nlp

modal run modal_app/prepare.py --smoke              # validate the path end to end
modal run modal_app/prepare.py                      # full alignment
modal run modal_app/train.py --stats-only           # measured unit durations + scores
modal run modal_app/train.py --inspect-only         # frame + class distribution
modal run modal_app/train.py --epochs 30 --patience 4 \
    --channels 48,96,192 --temporal-dim 384 --dilations 1,2,4,8,16 \
    --max-chunks-per-split 1500000 --run-name final

modal run modal_app/compare.py --runs a,b,c --n 100 # held-out Waxal, apples to apples
modal run modal_app/context_probe.py --run-name final  # acoustics vs context
modal volume get twi-phoneme checkpoints/final/best.pt .
```

`prepare.py` is idempotent: a shard whose manifest exists is skipped unless you
pass `--overwrite`, so a partial run resumes for free.

## How labels are made

1. Segment the transcript into grapheme units (`ghana_pico_asr.languages`).
2. CTC forced-align those units against the audio with
   [`ctc-forced-aligner`](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
   (MMS-300m, 20 ms emission frames). Its `preprocess_text` only splits on
   sentence/word/char, so the paired `(tokens_starred, text_starred)` lists are
   built directly — `<star>` wildcards at utterance edges and word boundaries,
   never between units inside a word, which would smear phoneme boundaries.
3. Compute log-mel and write per-shard blobs plus a unit/frame table.
4. Stage 2 derives a dense int16 label-per-frame track and cuts 1 s chunks at
   50% overlap.

Chunk length is only a batching convenience — utterances run 0.16 s to 30 s and
batching needs fixed shapes. The 50% overlap exists because frames near a chunk
edge have their receptive field clipped by zero padding, while at inference they
would see full context; overlapping gives every frame at least one interior view.

## Metric guidance

* **Validation/test UER is not comparable across runs.** Each run's val split
  reflects its own training mix, and it saturates at the aligner's label-noise
  floor — one run sat flat at ~0.40 from epoch 3 while balanced accuracy kept
  climbing 0.676 → 0.707.
* **Held-out Waxal is the only comparable number.** Use `modal_app/compare.py`.
* Checkpoint selection is on **balanced accuracy**, for the same reason.

## Long runs

`modal run` ties the app to a local client, so use `spawn` for anything long —
it queues the call server-side and the run becomes independent of this machine.

```bash
modal deploy modal_app/train.py            # once, and after any code change
python scripts/spawn_training.py --run-name xl --epochs 8 \
    --channels 96,192,384 --temporal-dim 640 --dilations 1,2,4,8,16

python scripts/spawn_training.py --status <call-id>
python scripts/spawn_training.py --list
modal app logs twi-phoneme-2dcnn
```

Call ids are recorded in `.spawned_runs.json` with their config. Log *tailing*
still depends on the local machine; losing it costs visibility, not the run.

## Preemption

A10G containers are preemptible. The trainer writes `last.pt` every epoch with
optimiser and scheduler state, so a restarted container resumes from its last
epoch instead of starting over, and `run_training` requests
`memory=32768, cpu=8.0` with retries.

`best.pt` holds weights only, so a run interrupted before this existed can only
be re-run, not resumed.

`last.pt` is written *after* best-checkpoint selection, so the `best` and
`stale` it carries are the ones the resumed run selects and early-stops
against. Written before selection — as it was until the xl run — a resumed run
compares epoch N+1 against epoch N-1's best, which lets a worse epoch overwrite
`best.pt` and stops `stale` from ever advancing toward `patience`.

## Moving a run to another workspace

Modal volumes are workspace-local, so a run cannot simply be repointed; but
nothing in `modal_app/` is workspace-specific either. Only the HF secret's name
differs, hence `PICO_HF_SECRET` (see `modal_app/common.py`).

Of the ~25 GB store, only two pieces need to travel — `modal_app/migrate.py`
does both:

| what | route | why |
|---|---|---|
| `features/mel`, `features/manifest`, `vocab.json` | public HF dataset, datacenter to datacenter | 24.1 GB; CPU containers only, so no GPU credits and no home uplink |
| `checkpoints/<run>/last.pt` (+`best.pt`) | through your machine | ~190 MB, cannot be regenerated, and belongs in no public dataset |
| `features/index` | neither | a cache the trainer rebuilds from the manifests |

The index rebuild is deterministic — `list_shards()` sorts and
`_cap_per_split()` seeds `default_rng(0)` — so the reconstructed chunk index and
train/val/test split are identical and validation stays comparable across the
move. Copying it would spend bandwidth to arrive at the same bytes;
`which_index` will name its artifacts if you want to anyway.

```bash
export MODAL_PROFILE=ghana-nlp                     # source
modal deploy modal_app/migrate.py
modal run modal_app/migrate.py::push --repo-id <org>/<dataset>

export MODAL_PROFILE=michseth PICO_HF_SECRET=huggingface-secret   # target
modal deploy modal_app/migrate.py modal_app/train.py
modal run modal_app/migrate.py::pull --repo-id <org>/<dataset>
modal volume put twi-phoneme ./hop/last.pt checkpoints/<run>/last.pt
python scripts/spawn_training.py --run-name <run> ...    # resumes at epoch N+1
```

Two things that bite:

* **`migrate.py` runs under its own app name.** Every other module shares
  `twi-phoneme-2dcnn`, so deploying one of them replaces the deployment — and a
  running call whose container is preempted retries into whatever is deployed
  then. Deploying a migration alongside a live run would turn a preemption into
  a dead run.
* **Verify transfers by size, never by exit status.** `modal volume get` on a
  directory has exited 0 having written truncated files (an 83 MB `chunks_*.npz`
  arrived as 0 bytes). Check the checkpoint by loading it and reading
  `history`.

Cancel the source call only once the target run has logged `[resume] picking up
from epoch N`, so a failure in the new workspace does not leave no run at all.

## Training on Hugging Face Jobs

An alternative to Modal that bills against HF credits. Spaces are the wrong
primitive here -- they host a long-running web app, not a batch job -- but
`hf jobs` is a direct analogue of `modal run`/`spawn`, with `--detach`,
`hf jobs logs`, `hf jobs cancel` and per-second GPU billing.

```bash
python scripts/launch_hf_job.py --smoke --flavor l40sx1     # validate a flavor
python scripts/launch_hf_job.py --run-name hf30 --epochs 30 --patience 0
hf jobs logs <job-id> --namespace ghananlpcommunity
```

The one structural difference is storage. Modal gives the trainer a single
writable volume; a Job gets three mounts, and `job/hf_train.py` assembles the
`/data` root out of them:

| mount | holds | why |
|---|---|---|
| `hf://datasets/...` at `/store` (ro) | mel, manifests, vocab | the published feature store; read-only, so nothing can be written back into it |
| `hf://buckets/...` at `/work` (rw) | `checkpoints/<run>/` | survives the job, so a restarted job resumes from `last.pt` |
| container disk at `/scratch` | the label index | rebuilt deterministically in ~4 min, and the build writes 300+ small files, which does not belong on a network bucket |

Mel blobs are read at random across millions of chunks, which is the worst case
for a network-backed mount, so the store is copied to local disk first -- one
sequential read, then every training read is local. `--no-copy` skips that if
the mount turns out to be fast.

Billing goes to `--namespace`, not the user, and the code is shipped by syncing
a staging copy of the package into a Hub bucket (the checkout path contains a
space, which the `-v LOCAL:/MOUNT` spec cannot carry).

### Epoch budget and early stopping

`--patience 0` is the default for long runs, deliberately. The cosine schedule
is sized to the full epoch budget, so a run stopped at 20 of 30 sits at ~25% of
peak LR rather than annealed, and a shorter properly-annealed run usually beats
it. Watch the metrics and cancel by hand instead; `best.pt` still tracks the
best epoch throughout.

This is not hypothetical: the 8-epoch `xl` run ended with its LR at **1.14e-11**,
so its final flattening was the schedule annealing rather than the model
converging -- val loss was still falling, and val frame accuracy still sat above
train.

## Decode settings

Frame predictions flicker between neighbouring units, and under
repeat-collapsing each flicker becomes a spurious unit. Smoothing the
posteriors and enforcing a minimum run length fixes it at no training cost:

| smoothing | min run | UER | units emitted (ref 128) |
|---|---|---|---|
| 1 frame | 2 | 0.562 | 150 |
| 5 | 2 | 0.520 | 139 |
| 9 | 4 | 0.454 | 112 |

`smooth=9, min_frames=4` was optimal for the 4M-parameter models. Re-swept on
the 12M `hf30` model it is not: **`smooth=7, min_frames=4`** wins, confirmed on
the 400-utterance held-out set rather than on the 120-utterance sweep that
proposed it.

| model | smooth 9 | smooth 7 | length ratio |
|---|---|---|---|
| hf30 | 0.3089 | **0.3058** | 0.901 -> 0.928 |
| xl-ep7 | 0.3103 | **0.3079** | 0.898 -> 0.924 |

Worth ~0.003 UER (1%) for free, and it moves the length ratio toward 1.0. The
lesson is that decode settings are model-specific: re-sweep after a capacity
change, and validate the winner on the larger set, because the sweep's own
margin (0.002 over 120 utterances) is inside its noise.

The remaining ~0.93 length ratio is not worth tuning away -- looser filters
close the gap but raise UER.

`--split-long-runs` re-splits a long run into repeated units to recover
geminates. It defaults **off**: it degrades UER on real audio.

## Corpora, measured (not advertised) figures

| corpus | utts aligned | audio | mean align score |
|---|---|---|---|
| `new-twi-tts-aligned` | 42,488 | 45.5 h | -0.70 |
| `twi-health-asr-gemini-500hrs` | 42,993 | 358.4 h | -1.07 |
| `kumawood-speech-transcriptions` | 158,088 | 214.0 h | -1.06 |
| `ghana-female-twi-asr-16word-splits` | 20,237 | 28.2 h | -0.93 |
| `twi-agriculture-speech` | 13,608 | 113.4 h | -0.69 |
| `asante-twi-bible-speech-text` | 33,079 | 63.8 h | **-0.30** |
| **combined** | **294,005** | **805.0 h** | |

After the utterance filter and label track: 294,005 utts / 805.0 h / 4.15 M
chunks, of which 482.1 h (59.9%) carries a usable frame label. `multispk`
(source id 6) is registered but absent — every shard crashed the audio decoder.

Kumawood aligns as well as the health corpus despite being Google-STT
transcribed, most likely because its segments average ~5 s rather than a flat
30 s — the aligner's free `<star>` has less room to absorb audio in a short
utterance. A first probe suggested -1.17, but that sampled only the first 300
rows per shard; films are ordered, so early segments (credits, intros, music)
are systematically worse than the body. **Sample across a shard, not from its
head.**

## Follow-ups

### 1. Cache key is too coarse (minor, costs ~2 min per run)

`ChunkConfig.key()` keys both the frame label track and the chunk index, but
`max_chunks_per_split` only affects the *index* subsampling. Changing it
invalidates a still-valid label track and forces a re-read of ~17 GB of mel
for the energy check.

Fix: split into two keys —
* label track: `min_unit_score`, `min_utt_score`, `min_unit_count`,
  `silence_as_ignore`, `min_silence_gap_ms`, `silence_energy_percentile`, `splits`
* chunk index: the above plus `chunk_ms`, `chunk_stride_ms`, `max_chunks_per_split`

### 2. Native digraphs align worse than English letters

Weighted share of units dropped at `min_unit_score=-1.0`:

```
ky 54.3%   dw 53.8%      <- native Twi digraphs
c  68.6%   x  65.3%      <- English /k~s/ and /ks/, no 1:1 phone
j  17.7%   v  18.6%   z 17.7%   <- English but phonemically transparent
native Twi (weighted)  27.9%
English    (weighted)  57.2%
```

MMS-300m has no real representation for /tɕ/, /dʑʷ/ etc., so it places 2-char
Twi units poorly and we lose ~half their data — exactly the units that justify
the digraph approach. They still learn well (`ky` 0.610, `dw` 0.742 recall), but
improving *alignment* for digraphs is the highest-value data fix available.

### 3. English code-switching is not handled, only filtered

No English digraphs are in the inventory (`ch sh th ph ck qu ng` all split), and
Twi digraph rules misfire on English (`twelve` -> `tw·e·l·v·e`, treating `tw` as
/tɕʷ/). The score filter discards the worst of it, so the failure mode is
benign — excluded rather than mislabelled. Fixing it properly needs per-word
language ID plus an English G2P. Not worth it for a Twi phoneme recogniser;
revisit if code-switched English becomes a target.
