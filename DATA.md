# Data

Interleaved-LM combines text, rendered text, and speech in one time-major sequence. Speech
preparation, text preparation, rendering, interleaving, and evaluation conversion are all included.

Dataset licenses apply to the data. Apache-2.0 covers only this repository's code.

## Model input

A batch contains `tokens[B, S + 1, C]`, where `C = 1 + K_image + K_audio`, and one input and
prediction mask per modality. The loader removes the first position from prediction masks to
align them with next-token targets.

Base and LF²AR use alternating interleaving. One modality is active at each position and the
other columns contain their pad value:

```text
time       0        1        2        3        4        5
modality   text     text     switch   audio    audio    audio
text       713      42       <swt>    <pad>    <pad>    <pad>
audio[0]   <pad>    <pad>    <pad>    91       91       204
audio[1]   <pad>    <pad>    <pad>    17       63       88
```

`<swt>` marks each modality change. Alignment durations map text tokens to modality frames and
are used to assemble the alternating chunks.

The Moshi-like pretraining layout is frame-synchronous. Each frame contains all audio codebooks and an aligned text token; `<pad>` fills frames between words, while `<epad>` marks word boundaries. The temporal backbone predicts the aligned text stream, while the depth transformer predicts the audio codebooks autoregressively, conditioned on the temporal representation and aligned text token.

```text
time       0        1        2        3
text       The      <pad>    dog      <epad>
audio[0]   91       91       204      17
audio[1]   17       63       88       42
...        ...      ...      ...      ...
audio[7]   203      11       74       90
```

### Modality value formats

The packed files use an integer carrier for every representation. `img_head_type` and
`aud_head_type` determine its model interpretation.

| Head | Stored row `[K]` | Model input and target | Loss / generation |
|---|---|---|---|
| `categorical` | IDs in `0 … vocab_size-1`, plus pad | One categorical value per codebook; optional delay pattern for `K > 1` | Cross-entropy; autoregressive sampling per codebook |
| `bernoulli` | Quantized intensities | Values are thresholded to bits; training inputs are Bernoulli samples from normalized intensities | Binary cross-entropy; bit sampling |
| `flow` | Quantized continuous values | Values are mapped to `[0, 1]`; all `K` channels form one continuous vector | Flow-matching MSE; Euler probability-flow sampling |

Padding has loss semantics only for categorical heads. Bernoulli and flow positions are selected
solely by their modality prediction mask. Multi-codebook categorical streams may use
`*_delay_pattern`; Bernoulli and flow heads model the complete `K`-dimensional row jointly.

For `configs/moshi-speech/2b.json`, prepare aligned eight-codebook Mimi shards under
`data/<dataset>/mimi/<split>/` using the recipe below.

## Packed shards

Every shard is a directory with `meta.json`, `samples.jsonl`, `skips.jsonl`, and SHA-256
checksums. Unknown schema versions are rejected.

| Stream | Required arrays | Shape and dtype |
|---|---|---|
| Text | `text.bin`, `text.len` | flat `int32` tokens; `int64` sample lengths |
| Rendered text | `image.bin`, `image.len`, `text.bin`, `text.len` | image `[sum_time, K]` `int16`; aligned text `int32` |
| Speech (`units-only`) | `audio.bin`, `audio.len` | units `[sum_time, K]` `int16`; lengths `int64` |
| Speech (`aligned`) | speech arrays plus `text.bin`, `text.len`, `dur.bin` | text `int32`; positive durations `int32` |

For every aligned sample, `sum(durations) == audio_length` (or rendered-frame length). Metadata
records source and preprocessing revisions, vocabulary and special-token IDs, counts, and file
checksums. `samples.jsonl` preserves stable IDs and source provenance; `skips.jsonl` records
machine-readable preparation failures.

## Training datasets

| Stream | Public source | Prepared output |
|---|---|---|
| Text | `HuggingFaceTB/smollm-corpus`: `fineweb-edu-dedup`, `cosmopedia-v2`, `python-edu` | `data/<alias>/smollm/<split>/<shard>/` |
| Rendered text | `HuggingFaceFW/finewiki`, config `en` | `data/finewiki/pixel/<split>/<shard>/` |
| Speech | LibriSpeech, LibriLight, Spoken Wikipedia, TED-LIUM 3, People's Speech, VoxPopuli, sTinyStories | `data/<alias>/hubert/<split>/<shard>/` |
| Aligned speech–text | LibriHeavy, Spoken Wikipedia, sTinyStories | same speech schema with aligned columns |

### Speech

Hours and unit counts below are the paper statistics. Acquire every corpus under its own terms
and preserve the official split or a recorded deterministic split.

| Source | Paper scale | Public source | Adapter |
|---|---:|---|---|
| LibriSpeech | 960 h / 67M | [OpenSLR 12](https://www.openslr.org/12), CC BY 4.0 | `manifest-librispeech` |
| LibriLight | 53k h / 3.74B | [LibriLight](https://github.com/facebookresearch/libri-light/tree/main/data_preparation), derived from LibriVox | `manifest-tsv` |
| Spoken Wikipedia | 1k h / 32M | [Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/), CC BY-SA 4.0 | `manifest-jsonl` |
| TED-LIUM 3 | 1.6k h / 110M | [OpenSLR 51](https://www.openslr.org/51), CC BY-NC-ND | `manifest-tsv` |
| People's Speech | 7k h / 480M | [`MLCommons/peoples_speech`](https://huggingface.co/datasets/MLCommons/peoples_speech); retain per-item licenses | `manifest-jsonl` |
| VoxPopuli English | 24k h / 1.64B | [`facebook/voxpopuli`](https://huggingface.co/datasets/facebook/voxpopuli) | `manifest-jsonl` |
| sTinyStories | 72k h / 4.82B | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories); historical audio used FastSpeech2/LJSpeech | Kokoro recipe below produces a distinct dataset |
| LibriHeavy | alignment source | [LibriHeavy](https://github.com/k2-fsa/libriheavy) | convert released cuts and word timestamps with `manifest-jsonl` |

Download and verify the TWIST layer-11 quantizer once:

```bash
mkdir -p assets
curl -L \
  https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin \
  -o assets/mhubert-25hz-l11-kmeans500.bin
echo '03cc04a9c24fec4285e73e709c485756d8f116aa8e724eac555de6a7cf8d28ad  assets/mhubert-25hz-l11-kmeans500.bin' \
  | sha256sum --check
```

Create one manifest record per segment. `words` are required only for aligned preparation; use
`--alignment-source mms` with a transcript when corpus timestamps are unavailable.

```json
{"id":"source:train:0001","dataset":"source","split":"train","audio":"/data/0001.flac","offset":0.0,"duration":4.2,"sample_rate":16000,"transcript":"hello world","source_revision":"release-tag","words":[{"text":"hello","start":0.1,"end":0.5},{"text":"world","start":0.6,"end":1.0}]}
```

```bash
interleaved-speech-data manifest-jsonl \
  --input acquired.jsonl --output manifests/source.jsonl
interleaved-speech-data plan \
  --manifest manifests/source.jsonl --num-shards 128 \
  --output manifests/source.plan.json
interleaved-speech-data encode \
  --manifest manifests/source.jsonl --plan manifests/source.plan.json \
  --shard-index "$SHARD" --output data/source/hubert/train \
  --speech-tokenizer hubert \
  --profile aligned --text-tokenizer HuggingFaceTB/SmolLM-360M \
  --text-tokenizer-revision 59f7ef243ee09a72cbc14cb054393a3e3b771d41 \
  --hubert-model slprl/mhubert-base-25hz \
  --hubert-revision a319086e1d343190047d02b7f81133fb310c1b90 \
  --kmeans assets/mhubert-25hz-l11-kmeans500.bin --no-hubert-normalize
```

Use `--profile units-only` for unaligned streams. Aligned shards place word spans on the
uncollapsed 25 Hz timeline, map them to collapsed runs, and require positive durations covering
the complete stored sequence. Invalid samples are written to `skips.jsonl`.

### Mimi

Moshi-like pretraining uses the official Mimi tokenizer: eight 2,048-entry residual codebooks at
12.5 Hz from 24 kHz mono audio. The first codebook is semantic; the remaining seven are acoustic.
The packed vocabulary adds ID 2,048 as the sample-boundary/padding value.

```bash
interleaved-speech-data encode \
  --manifest manifests/source.jsonl --plan manifests/source.plan.json \
  --shard-index "$SHARD" --output data/source/mimi/train \
  --speech-tokenizer mimi \
  --mimi-repo kyutai/moshiko-pytorch-bf16 \
  --mimi-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --profile aligned --text-tokenizer HuggingFaceTB/SmolLM-360M \
  --text-tokenizer-revision 59f7ef243ee09a72cbc14cb054393a3e3b771d41 \
  --alignment-source timestamps
```

The Moshi pretraining configuration applies delay 0 to the semantic codebook and a uniform delay
of 2 frames to all acoustic codebooks. It weights semantic-codebook loss by 100 and each acoustic
loss by 1. Aligned text conditions the eight within-frame Mimi predictions through a 2-layer,
256-dimensional depth transformer. A learned 2,048-to-256 projection separates it from the
temporal backbone, and each codebook has its own 256-dimensional output head.

### Text

Run each recoverable SmolLM component independently:

```bash
interleaved-text-data \
  --dataset HuggingFaceTB/smollm-corpus --name fineweb-edu-dedup \
  --dataset-revision 413c820f4b4875ebb8f3e90db0f7474c9dc0c5d4 \
  --tokenizer-id HuggingFaceTB/SmolLM-360M \
  --tokenizer-revision 59f7ef243ee09a72cbc14cb054393a3e3b771d41 \
  --output data/finewebedu/smollm --num-shards 256 --shard-index "$SHARD"
```

Use `cosmopedia-v2` → `data/cosmopedia2/smollm` and `python-edu` →
`data/pythonedu/smollm` for the other components.

### Rendered text

The paper renderer uses Go Noto Current, 8-point text at 120 DPI, 16×16 patches, black glyphs on
white, and keeps bigrams inside word boundaries.

```bash
python renderer/download_font.py renderer
interleaved-render \
  --dataset HuggingFaceFW/finewiki --name en \
  --dataset-revision 71306a3fed631b8c1a77e5479eb9edddc8866003 \
  --split train --text-column text \
  --tokenizer-id HuggingFaceTB/SmolLM-360M \
  --tokenizer-revision 59f7ef243ee09a72cbc14cb054393a3e3b771d41 \
  --renderer-dir renderer --output data/finewiki/pixel \
  --num-shards 256 --shard-index "$SHARD"
```

Rendered TinyStories uses the same command with `roneneldan/TinyStories` at revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

### TinyStories speech

The paper's sTinyStories used FastSpeech2 with the LJSpeech voice. The maintained public recipe
uses Kokoro and is deliberately named `tinystories-kokoro`; it is not bit-identical to the paper
data. It initializes Kokoro once per worker and can discard waveforms immediately after unit
extraction. HuBERT preparation resamples the 24 kHz synthesis to 16 kHz; Mimi consumes it directly.

```bash
interleaved-speech-data manifest-tinystories \
  --dataset roneneldan/TinyStories \
  --dataset-revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --split train --output manifests/tinystories.jsonl
interleaved-speech-data plan --manifest manifests/tinystories.jsonl --num-shards 256 \
  --output manifests/tinystories.plan.json
interleaved-speech-data tinystories \
  --manifest manifests/tinystories.jsonl --plan manifests/tinystories.plan.json \
  --shard-index "$SHARD" --output data/tinystories-kokoro/hubert/train \
  --dataset-revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --kokoro-revision f3ff3571791e39611d31c381e3a41a3af07b4987 \
  --speech-tokenizer hubert \
  --profile aligned --text-tokenizer HuggingFaceTB/SmolLM-360M \
  --text-tokenizer-revision 59f7ef243ee09a72cbc14cb054393a3e3b771d41 \
  --hubert-model slprl/mhubert-base-25hz \
  --hubert-revision a319086e1d343190047d02b7f81133fb310c1b90 \
  --kmeans assets/mhubert-25hz-l11-kmeans500.bin --no-hubert-normalize
```

For the Mimi version, write to `data/tinystories-kokoro/mimi/train`, select
`--speech-tokenizer mimi`, and pass the pinned `--mimi-repo` and `--mimi-revision` from the Mimi
recipe above.

## Evaluation data

| Task | Source and conversion |
|---|---|
| MMLU | `cais/mmlu`; evaluator materializes the selected modality and records the revision |
| HellaSwag | `hellaswag`; same provenance rules |
| sStoryCloze | released 1,871-question semantic task; `interleaved-storycloze --task sstorycloze` |
| tStoryCloze | released 1,871-question temporal-distractor task; `interleaved-storycloze --task tstorycloze` |

```bash
interleaved-storycloze \
  --task sstorycloze --input acquired/sstorycloze.csv --output data
```

The converter validates complete choice groups and writes `manifest.jsonl` and
`manifest.meta.json`. Pass its source checksum to `interleaved-eval --dataset_revision`.
