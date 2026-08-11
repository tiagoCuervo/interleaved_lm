# Multimodal Interleaved Language Modeling

[![COLM 2026](https://img.shields.io/badge/COLM-2026-6f42c1.svg)](https://openreview.net/forum?id=gsM9d0iZqf)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-FFD21E.svg)](https://huggingface.co/tiagoCuervo/lf2ar-speech-360m)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Minimal research codebase a la [nanoGPT](https://github.com/karpathy/nanogpt) for language models that interleave text with token- or vector-valued modalities. The code accompanies [*LF²AR: Accounting for Layerwise Dynamics to Improve Multimodal Adaptation of Language Models*](https://openreview.net/forum?id=gsM9d0iZqf). The paper focuses on rendered text and speech, but the framework supports modalities represented by categorical codebooks, binary features, or continuous vectors trained with flow matching (this last one not heavily validated). We provide configurations for the LF²AR models presented in the paper, alongside [SpiritLM](https://arxiv.org/abs/2402.05755)- and [Moshi](https://arxiv.org/abs/2410.00037)-style pretraining configurations implemented within the same minimal framework.

See [DATA.md](DATA.md) for data formats and preparation, and
[RESULTS.md](RESULTS.md) for expected checkpoint scores. External model and font credits are in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Install

Python 3.11 and PyTorch 2.4+ are supported.

```bash
conda create -n interleaved-lm python=3.11 -y
conda activate interleaved-lm
pip install -c requirements/constraints.txt -e '.[evaluation,speech-data,mimi,audio,notebook]'
```

Rendered-text preparation additionally requires Cairo and Pango, then
`pip install -e '.[rendered-text]'`.

Run the test suite to verify the installation:

```bash
pytest -q
```

## Data

The paper uses three equally sampled streams: text, speech, and aligned speech–text. The speech
mixture contains LibriSpeech, LibriLight, Spoken Wikipedia, TED-LIUM 3, People's Speech,
VoxPopuli, and sTinyStories; aligned data uses LibriHeavy, Spoken Wikipedia, and sTinyStories.
Text comes from the recoverable SmolLM corpus components. Pixel experiments render FineWiki.

```bash
# Text shard
interleaved-text-data \
  --dataset HuggingFaceTB/smollm-corpus --name fineweb-edu-dedup \
  --dataset-revision 413c820f4b4875ebb8f3e90db0f7474c9dc0c5d4 \
  --tokenizer-id HuggingFaceTB/SmolLM-360M \
  --tokenizer-revision 59f7ef243ee09a72cbc14cb054393a3e3b771d41 \
  --output data/finewebedu/smollm --num-shards 256 --shard-index "$SHARD"

# Rendered FineWiki shard
python renderer/download_font.py renderer
interleaved-render \
  --dataset HuggingFaceFW/finewiki --name en \
  --dataset-revision 71306a3fed631b8c1a77e5479eb9edddc8866003 \
  --split train --text-column text --renderer-dir renderer \
  --tokenizer-id HuggingFaceTB/SmolLM-360M \
  --tokenizer-revision 59f7ef243ee09a72cbc14cb054393a3e3b771d41 \
  --output data/finewiki/pixel --num-shards 256 --shard-index "$SHARD"
```

Speech preparation is built in: `interleaved-speech-data` writes units-only or aligned shards.
[DATA.md](DATA.md) gives every public source, revision, output layout, head-specific
representation, alignment invariant, and evaluation converter.

## Train LF²AR

The supplied speech configurations perform one uninterrupted joint training run from a
pretrained SmolLM backbone. Each uses context 2,048 and exactly 983,040 tokens per update; the
paper rounds this batch to 1M tokens.

| Config | Parameters | Backbone | Updates | Tokens | LR | Attention residual | Stable A100-80 MFU |
|---|---:|---|---:|---:|---:|---|---:|
| `configs/lf2ar-speech/150m.json` | 149M | SmolLM-135M | 17k | 16.7B | 3e-4 | dynamic | 36.79% |
| `configs/lf2ar-speech/360m.json` | 402M | SmolLM-360M | 17k | 16.7B | 3e-4 | dynamic | 42.75% |
| `configs/lf2ar-speech/2b.json` | 1.98B | SmolLM-1.7B | 33k | 32.4B | 1e-4 | dynamic | 41.43% |

MFU is the median raw value over controlled full optimizer updates on one A100 80GB PCIe after compilation and optimizer warmup. 

```bash
# DDP
torchrun --standalone --nproc-per-node=8 -m interleaved_lm.train \
  configs/lf2ar-speech/360m.json

# FSDP (the 2B config selects it by default)
torchrun --standalone --nproc-per-node=8 -m interleaved_lm.train \
  configs/lf2ar-speech/2b.json
```

Parameters and optimizer states remain FP32; BF16 is autocast. `gradient_accumulation_steps` is
the global accumulation count and must be divisible by world size.

## Train architecture baselines

| Config | Layout | Parameters | Audio representation | Auxiliary decoder | Stable A100-80 MFU |
|---|---|---:|---|---|---:|
| `configs/base-speech/2b.json` | Spirit LM-like interleaving | 1.71B | one mHuBERT codebook | none | 59.00% |
| `configs/moshi-speech/2b.json` | synchronized text + speech | 1.75B | eight Mimi codebooks | 2 layers, width 256 | 55.36% |

Both initialize the SmolLM-1.7B temporal backbone and keep the same 983,040-token global update.
The Moshi-like recipe is pretraining-only: aligned text with `<pad>`/`<epad>` conditions the depth
transformer, the semantic stream has zero delay, and all seven acoustic streams have delay 2.
It does not implement full-duplex dialogue.

```bash
torchrun --standalone --nproc-per-node=8 -m interleaved_lm.train configs/base-speech/2b.json
torchrun --standalone --nproc-per-node=8 -m interleaved_lm.train configs/moshi-speech/2b.json
```

## Evaluate

`interleaved-eval` scores MMLU, HellaSwag, sStoryCloze, and tStoryCloze for supported input and
output modalities. It accepts either a training checkpoint or a native model directory.

```bash
interleaved-eval \
  --checkpoint tiagoCuervo/lf2ar-speech-360m \
  --eval_dataset sstorycloze \
  --dataset_revision SHA256_FROM_MANIFEST_METADATA \
  --in_modality audio --out_modality text \
  --data_root data --hubert_model slprl/mhubert-base-25hz \
  --hubert_revision a319086e1d343190047d02b7f81133fb310c1b90 \
  --hubert_km_path assets/mhubert-25hz-l11-kmeans500.bin \
  --batch_size 16 --dtype bfloat16 \
  --output results/sstorycloze-audio-text.json
```

Expected StoryCloze accuracies for all 150M, 400M, and 2B LF²AR modality directions are in
[RESULTS.md](RESULTS.md).

## Load and generate

```python
from interleaved_lm import PerceptionExpressionAdaptedTextLM
from interleaved_lm.audio import generate_speech

model = PerceptionExpressionAdaptedTextLM.from_pretrained(
    "tiagoCuervo/lf2ar-speech-360m", device="cuda"
)
codes = generate_speech(model, "A short story about the moon", max_new_tokens=250)
```

[notebooks/generate_audio.ipynb](notebooks/generate_audio.ipynb) visualizes and plays aligned
interleaved samples, then loads a trained or Hugging Face model, generates units, and decodes WAV
audio. Convert a training checkpoint with:

```bash
interleaved-convert out/run/checkpoint-00017000.pt out/run/native
```

## Citation

```bibtex
@inproceedings{cuervo2026accounting,
  title     = {{LF}${}^{2}${AR}: Accounting for Layerwise Dynamics to Improve Multimodal Adaptation of Language Models},
  author    = {Cuervo, Santiago and Moumen, Adel and Labrak, Yanis and Khurana, Sameer and Laurent, Antoine and Rouvier, Mickael and Woodland, Phil and Marxer, Ricard},
  booktitle = {Third Conference on Language Modeling},
  year      = {2026},
  url       = {https://openreview.net/forum?id=gsM9d0iZqf}
}
```
