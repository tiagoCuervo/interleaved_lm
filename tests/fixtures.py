from __future__ import annotations

from pathlib import Path

import numpy as np

from interleaved_speech_data.packing import PreparedSample, ShardWriter


def create_toy_dataset(
    output: str | Path,
    profile: str = "units-only",
    seed: int = 1337,
    n_codebooks: int = 1,
) -> None:
    """Create deterministic packed units for tests."""
    rng = np.random.default_rng(seed)
    output = Path(output)
    for split, count in (("train", 32), ("val", 8)):
        preprocessing = {
            "generator": "interleaved-speech-data-toy-v1",
            "seed": seed,
            "eos_token": 32,
            "pad_token": 33,
            "vocab_size": 34,
            "sample_rate": 16000,
        }
        if profile == "aligned":
            preprocessing.update(
                text_tokenizer="toy-tokenizer",
                text_base_vocab_size=100,
                text_eos_id=2,
                text_pad_id=100,
                text_end_of_word_id=101,
                text_special_vocab_size=102,
            )
        with ShardWriter(
            output / split,
            "shard-00000",
            profile=profile,
            n_codebooks=n_codebooks,
            preprocessing=preprocessing,
        ) as writer:
            if writer.already_complete:
                continue
            for index in range(count):
                length = int(rng.integers(48, 80))
                codes = rng.integers(
                    0, 32, size=(length - 1, n_codebooks), dtype=np.int16
                )
                codes = np.concatenate(
                    (codes, np.full((1, n_codebooks), 32, dtype=np.int16))
                )
                text = durations = None
                if profile == "aligned":
                    n_tokens = 16
                    text = rng.integers(0, 100, size=n_tokens, dtype=np.int32)
                    text[1::2] = 101
                    boundaries = np.linspace(0, length, n_tokens + 1, dtype=np.int32)
                    durations = np.diff(boundaries)
                writer.write(
                    PreparedSample(
                        id=f"toy:{split}:{index:04d}",
                        codes=codes,
                        text_tokens=text,
                        token_durations=durations,
                        metadata={"dataset": "toy", "split": split},
                    )
                )
