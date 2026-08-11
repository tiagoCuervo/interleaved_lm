import json
from pathlib import Path

import torch

from interleaved_lm import PerceptionExpressionAdaptedTextLM
from interleaved_lm.checkpoint import convert_checkpoint
from interleaved_lm.train import TrainSettings, _batch_provenance, _schedule, train
from tests.fixtures import create_toy_dataset


def _write_config(
    path: Path,
    data_root: Path,
    output: Path,
    *,
    max_steps: int,
    resume=None,
):
    training = {
        "output_dir": str(output),
        "resume": None if resume is None else str(resume),
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_steps": max_steps,
        "eval_interval": 1,
        "eval_steps": 1,
        "save_interval": 1,
        "log_interval": 1,
        "warmup_steps": 1,
        "warmdown_steps": 1,
        "precision": "float32",
        "compile": False,
    }
    payload = {
        "data_root": str(data_root),
        "train_data": {
            "aud_datasets": ["toy"],
            "splits": ["train"],
            "p_strategies": [0, 0, 1, 0, 0],
            "txt_tokens": "smollm",
            "img_tokens": "pixel",
            "aud_tokens": "hubert",
            "block_size": 16,
        },
        "validation_data": {
            "aud_datasets": ["toy"],
            "splits": ["val"],
            "p_strategies": [0, 0, 1, 0, 0],
            "txt_tokens": "smollm",
            "img_tokens": "pixel",
            "aud_tokens": "hubert",
            "block_size": 16,
        },
        "model": {
            "backbone": "local-test",
            "backbone_config": {
                "model_type": "llama",
                "vocab_size": 34,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 32,
                "tie_word_embeddings": True,
            },
            "warm_init": False,
            "aud_inadapter_n_layers": 0,
            "aud_inadapter_dim": 32,
            "aud_inadapter_mlp_dim": 64,
            "aud_inadapter_n_heads": 4,
            "aud_inadapter_n_kvheads": 2,
            "aud_outadapter_n_layers": 0,
            "aud_outadapter_dim": 32,
            "aud_outadapter_mlp_dim": 64,
            "aud_outadapter_n_heads": 4,
            "aud_outadapter_n_kvheads": 2,
        },
        "training": training,
    }
    path.write_text(json.dumps(payload))


def test_cpu_toy_training_resume_export_and_generate(tmp_path):
    data_root = tmp_path / "data"
    create_toy_dataset(data_root / "toy" / "hubert")
    output = tmp_path / "out"
    config = tmp_path / "train.json"
    _write_config(config, data_root, output, max_steps=3)
    train(config)

    checkpoint = output / "checkpoint-00000002.pt"
    assert checkpoint.exists()
    resumed_output = tmp_path / "resumed"
    _write_config(config, data_root, resumed_output, max_steps=3, resume=checkpoint)
    train(config)

    resumed = resumed_output / "checkpoint-00000003.pt"
    baseline_state = torch.load(
        output / "checkpoint-00000003.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    resumed_state = torch.load(
        resumed, map_location="cpu", weights_only=False
    )["model"]
    assert baseline_state.keys() == resumed_state.keys()
    for name in baseline_state:
        torch.testing.assert_close(
            baseline_state[name], resumed_state[name], rtol=0, atol=0
        )
    export = tmp_path / "native"
    convert_checkpoint(resumed, export)
    model = PerceptionExpressionAdaptedTextLM.from_pretrained(export).eval()
    prompt = torch.randint(0, 32, (1, 4, 1))
    empty = torch.zeros((1, 4), dtype=torch.bool)
    audio = torch.ones((1, 4), dtype=torch.bool)
    generated = model.generate(
        prompt,
        empty,
        empty,
        audio,
        max_new_tokens=2,
        gen_aud=True,
        temperature=0,
    )
    assert generated.shape[1] == 6


def test_approximately_one_million_means_exact_batch_math():
    settings = TrainSettings(batch_size=20, gradient_accumulation_steps=24)
    provenance = _batch_provenance(settings, context_tokens=2048, world_size=8)
    assert provenance["global_sequences_per_update"] == 480
    assert provenance["tokens_per_update"] == 983_040


def test_warmdown_reaches_floor_on_final_update():
    settings = TrainSettings(max_steps=10, warmup_steps=2, warmdown_steps=3)
    assert _schedule(6, 2e-4, 2e-5, settings) == 2e-4
    assert _schedule(9, 2e-4, 2e-5, settings) == 2e-5
