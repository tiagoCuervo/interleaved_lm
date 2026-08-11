# Third-party notices

## Text backbone and renderer

Interleaved-LM initializes its text backbone and tokenizer from the Apache-2.0
[SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
135M, 360M, and 1.7B releases by Hugging Face.

Rendered-text preparation downloads `GoNotoCurrent.ttf` from the pinned Apache-2.0
[PIXEL artifact](https://huggingface.co/Team-PIXEL/pixel-base-bigrams). The merged Noto font
retains the terms of its constituent fonts and is not redistributed here.

## Speech tokenizer and vocoder

The 25 Hz mHuBERT checkpoint, layer-11 K-means quantizer, and CodeHiFiGAN checkpoint used by
the published recipes are from Meta's
[textlesslib](https://github.com/facebookresearch/textlesslib) and its
[TWIST assets](https://github.com/facebookresearch/textlesslib/tree/main/examples/twist).
The Hugging Face mHuBERT conversion is maintained by
[SLP-RL](https://huggingface.co/slprl/mhubert-base-25hz). Please cite both textlesslib and
TWIST when using these artifacts:

```bibtex
@article{kharitonov2022textless,
  title   = {textless-lib: A Library for Textless Spoken Language Processing},
  author  = {Kharitonov, Eugene and Copet, Jade and Lakhotia, Kushal and Nguyen, Tu Anh and Tomasello, Paden and Lee, Ann and Elkahky, Ali and Hsu, Wei-Ning and Mohamed, Abdelrahman and Dupoux, Emmanuel and Adi, Yossi},
  journal = {arXiv preprint arXiv:2202.07359},
  year    = {2022}
}

@inproceedings{hassid2023textually,
  title     = {Textually Pretrained Speech Language Models},
  author    = {Hassid, Michael and Remez, Tal and Nguyen, Tu Anh and Gat, Itai and Conneau, Alexis and Kreuk, Felix and Copet, Jade and Defossez, Alexandre and Synnaeve, Gabriel and Dupoux, Emmanuel and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```

`src/interleaved_lm/vocoder.py` contains a compact inference-only adaptation of the CodeHiFiGAN generator
from textlesslib and the original
[HiFi-GAN](https://github.com/jik876/hifi-gan). Both upstream implementations are MIT licensed.
Their trained checkpoint and configuration are downloaded from Meta's public artifact host and
verified against the SHA-256 values recorded in textlesslib. Model weights retain their upstream
terms and are not part of this repository's Apache-2.0 license.

The optional eight-codebook pipeline uses Kyutai's
[Mimi implementation](https://github.com/kyutai-labs/moshi) and the Mimi checkpoint distributed
with [`kyutai/moshiko-pytorch-bf16`](https://huggingface.co/kyutai/moshiko-pytorch-bf16).
Kyutai's Python code is MIT licensed and the model weights are CC BY 4.0; neither is redistributed
by this repository.

## TinyStories synthesis

The optional synthesis recipe uses [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
and the [`kokoro`](https://github.com/hexgrad/kokoro) package by hexgrad. Kokoro code and model
weights are Apache-2.0 licensed and are downloaded at the pinned revision; they are not
redistributed here.

## Architecture references

The Base alternating speech/text layout follows
[Spirit LM](https://arxiv.org/abs/2402.05755). The synchronized text-plus-audio hierarchy,
delayed acoustic streams, and depth-transformer configuration in the Moshi-like pretraining
recipe follow [Moshi](https://arxiv.org/abs/2410.00037). This repository does not redistribute
either project's models or data.

## MIT notices

MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
