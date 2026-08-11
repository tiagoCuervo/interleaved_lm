# Expected results

StoryCloze accuracy is measured on 1,871 two-choice questions per task. `AA`, `TT`, `AT`, and
`TA` denote audio→audio, text→text, audio→text, and text→audio.

| Model | s-AA | s-TT | s-AT | s-TA | t-AA | t-TT | t-AT | t-TA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 150M (148,983,612 parameters) | 55.16 | 64.08 | 58.26 | 58.47 | 81.99 | 88.24 | 80.60 | 75.31 |
| 400M (401,665,984 parameters) | 57.19 | 68.63 | 61.89 | 62.27 | 84.61 | 91.39 | 84.98 | 80.81 |
| 2B (1,980,911,664 parameters) | 61.57 | 73.70 | 63.98 | 64.03 | 87.60 | 92.52 | 86.91 | 83.97 |

## A100 MFU

| Model | Tokens/update | MFU |
|---:|---:|---:|
| 150M | 983,040 | 36.79% |
| 400M | 983,040 | 42.75% |
| 2B | 983,040 | 41.43% |
| Spirit LM-like 2B Base | 983,040 | 59.00% |
| Moshi-like 2B | 983,040 | 55.36% |

Each value is the median raw MFU over controlled full optimizer updates on one A100 80GB PCIe,
using each listed configuration's microbatch and accumulation schedule after compilation and
optimizer warmup. Measurements use BF16 autocast, FP32 parameters and AdamW
state, compilation, context 2,048, and 983,040 tokens per update. MFU uses the nanoGPT/PaLM
`6N + attention` estimate and a 312-TFLOP/s BF16 peak.
