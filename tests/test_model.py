import json
from pathlib import Path

import pytest
import torch

from interleaved_lm import ModelArgs, PerceptionExpressionAdaptedTextLM

from .helpers import tiny_backbone


def build_model(
    tmp_path,
    *,
    n_codebooks=1,
    head="categorical",
    attention_residual="dynamic",
    codebook_transformer=False,
    text_prefix=False,
    depth_dim=32,
):
    tokenizer, backbone_config = tiny_backbone(tmp_path)
    args = ModelArgs(
        txt_vocabsize=102,
        aud_vocabsize=34,
        txt_pad_token=100,
        aud_pad_token=33,
        swt_token=None if text_prefix else 101,
        predict_txt_special_tokens=text_prefix,
        txt_pad_loss_weight=0.5 if text_prefix else 1.0,
        n_aud_codebooks=n_codebooks,
        block_size=16,
        backbone="local-test",
        backbone_config=backbone_config,
        tokenizer=str(tokenizer),
        warm_init=False,
        aud_inadapter_n_layers=1,
        aud_inadapter_dim=32,
        aud_inadapter_mlp_dim=64,
        aud_inadapter_n_heads=4,
        aud_inadapter_n_kvheads=2,
        aud_outadapter_n_layers=1,
        aud_outadapter_dim=32,
        aud_outadapter_mlp_dim=64,
        aud_outadapter_n_heads=4,
        aud_outadapter_n_kvheads=2,
        aud_attention_residual=attention_residual,
        aud_head_type=head,
        aud_codebook_transformer_layers=1 if codebook_transformer else 0,
        aud_codebook_transformer_dim=depth_dim,
        aud_codebook_transformer_mlp_dim=2 * depth_dim,
        aud_codebook_transformer_n_heads=4,
        aud_codebook_transformer_n_kvheads=2,
        aud_codebook_transformer_text_prefix=text_prefix,
        tie_aud_embeddings=depth_dim == 32,
        aud_flow_steps=2,
        aud_flow_d=16,
    )
    return PerceptionExpressionAdaptedTextLM(args, is_resume=True)


def build_image_model(tmp_path, *, head="categorical"):
    tokenizer, backbone_config = tiny_backbone(tmp_path)
    args = ModelArgs(
        txt_vocabsize=102,
        img_vocabsize=17,
        txt_pad_token=100,
        img_pad_token=16,
        swt_token=101,
        n_img_codebooks=2,
        block_size=16,
        backbone="local-test",
        backbone_config=backbone_config,
        tokenizer=str(tokenizer),
        warm_init=False,
        img_inadapter_n_layers=1,
        img_inadapter_dim=24,
        img_inadapter_mlp_dim=48,
        img_inadapter_n_heads=4,
        img_inadapter_n_kvheads=2,
        img_outadapter_n_layers=1,
        img_outadapter_dim=32,
        img_outadapter_mlp_dim=64,
        img_outadapter_n_heads=4,
        img_outadapter_n_kvheads=2,
        img_attention_residual="dynamic",
        img_head_type=head,
        img_flow_steps=2,
        img_flow_d=16,
        tie_img_embeddings=False,
    )
    return PerceptionExpressionAdaptedTextLM(args, is_resume=True)


@pytest.mark.parametrize("head,n_codebooks", [("categorical", 1), ("categorical", 2), ("flow", 2)])
def test_speech_forward_backward(tmp_path, head, n_codebooks):
    model = build_model(tmp_path, n_codebooks=n_codebooks, head=head)
    batch, steps = 2, 12
    tokens = torch.full((batch, steps + 1, 1 + n_codebooks), 100, dtype=torch.long)
    tokens[..., 1:] = torch.randint(0, 32, (batch, steps + 1, n_codebooks))
    txt_in = torch.zeros((batch, steps + 1), dtype=torch.bool)
    aud_in = torch.ones_like(txt_in)
    img_in = torch.zeros_like(txt_in)
    pred = torch.ones((batch, steps), dtype=torch.bool)
    empty = torch.zeros_like(pred)
    model(tokens, txt_in, img_in, aud_in, empty, empty, pred)
    assert torch.isfinite(model.last_loss)
    model.last_loss.backward()


@pytest.mark.parametrize("head", ["categorical", "bernoulli", "flow"])
def test_rendered_text_forward_backward_and_generation(tmp_path, head):
    model = build_image_model(tmp_path, head=head)
    batch, steps = 2, 10
    tokens = torch.full((batch, steps + 1, 3), 100, dtype=torch.long)
    tokens[..., 1:] = torch.randint(0, 16, (batch, steps + 1, 2))
    empty_input = torch.zeros((batch, steps + 1), dtype=torch.bool)
    image_input = torch.ones_like(empty_input)
    empty_pred = torch.zeros((batch, steps), dtype=torch.bool)
    image_pred = torch.ones_like(empty_pred)
    _, image_logits, _ = model(
        tokens,
        empty_input,
        image_input,
        empty_input,
        empty_pred,
        image_pred,
        empty_pred,
    )
    assert image_logits.shape[0] == batch * steps
    assert torch.isfinite(model.last_loss)
    model.last_loss.backward()

    model.eval()
    prompt = tokens[:1, :4]
    image_prompt = torch.ones((1, 4), dtype=torch.bool)
    empty_prompt = torch.zeros_like(image_prompt)
    generated = model.generate(
        prompt,
        empty_prompt,
        image_prompt,
        empty_prompt,
        max_new_tokens=3,
        gen_img=True,
        temperature=0,
    )
    assert generated.shape == (1, 7, 3)
    assert generated[:, 4:, 1:].min() >= 0
    assert generated[:, 4:, 1:].max() < 17


def test_native_roundtrip(tmp_path):
    model = build_model(tmp_path / "source").eval()
    export = tmp_path / "export"
    model.save_pretrained(export, max_shard_size=20_000)
    restored = PerceptionExpressionAdaptedTextLM.from_pretrained(export)
    assert set(model.state_dict()) == set(restored.state_dict())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.float(), restored.state_dict()[name].float())


def test_audio_only_model_needs_no_tokenizer(tmp_path):
    _, backbone_config = tiny_backbone(tmp_path)
    args = ModelArgs(
        txt_vocabsize=None,
        aud_vocabsize=34,
        aud_pad_token=33,
        n_aud_codebooks=1,
        block_size=16,
        backbone="local-test",
        backbone_config=backbone_config,
        warm_init=False,
        aud_inadapter_n_layers=1,
        aud_inadapter_dim=32,
        aud_inadapter_mlp_dim=64,
        aud_inadapter_n_heads=4,
        aud_inadapter_n_kvheads=2,
        aud_outadapter_n_layers=1,
        aud_outadapter_dim=32,
        aud_outadapter_mlp_dim=64,
        aud_outadapter_n_heads=4,
        aud_outadapter_n_kvheads=2,
    )
    model = PerceptionExpressionAdaptedTextLM(args, is_resume=True)
    export = tmp_path / "audio-only"
    model.save_pretrained(export)
    restored = PerceptionExpressionAdaptedTextLM.from_pretrained(export)
    assert restored.global_workspace.txt_tokenizer is None


def test_zero_temperature_is_greedy(tmp_path):
    model = build_model(tmp_path)
    logits = torch.tensor([[0.5, 3.0, 1.0]])
    assert model.sample_from_logits(logits, temperature=0).item() == 1


def test_delay_pattern_supports_long_compiled_delay(tmp_path):
    model = build_image_model(tmp_path)
    tokens = torch.arange(2 * 34 * 3).reshape(2, 34, 3)
    eager = model.apply_delay_pattern(tokens, 999, [0, 18, 40])
    compiled = torch.compile(model.apply_delay_pattern, backend="eager", fullgraph=True)(
        tokens, 999, [0, 18, 40]
    )
    torch.testing.assert_close(compiled, eager)
    torch.testing.assert_close(eager[:, :, 0], tokens[:, :, 0])
    assert torch.all(eager[:, :18, 1] == 999)
    torch.testing.assert_close(eager[:, 18:, 1], tokens[:, :16, 1])
    assert torch.all(eager[:, :, 2] == 999)


def test_cross_modal_generation_does_not_count_switch_as_output(tmp_path):
    model = build_model(tmp_path).eval()
    prompt = torch.tensor([[[2, 33], [3, 33], [4, 33]]])
    txt = torch.ones((1, 3), dtype=torch.bool)
    empty = torch.zeros_like(txt)
    output = model.generate(
        prompt,
        txt,
        empty,
        empty,
        max_new_tokens=2,
        gen_aud=True,
        temperature=0,
    )
    assert output.shape[1] == prompt.shape[1] + 1 + 2
    assert output[0, prompt.shape[1], 0] == model.swt_token


def test_zero_length_generation_does_not_insert_switch(tmp_path):
    model = build_model(tmp_path).eval()
    prompt = torch.tensor([[[2, 33], [3, 33], [4, 33]]])
    txt = torch.ones((1, 3), dtype=torch.bool)
    empty = torch.zeros_like(txt)
    output = model.generate(
        prompt,
        txt,
        empty,
        empty,
        max_new_tokens=0,
        gen_aud=True,
    )
    torch.testing.assert_close(output, prompt)


def test_depth_transformer_generation(tmp_path):
    model = build_model(
        tmp_path, n_codebooks=2, codebook_transformer=True
    ).eval()
    prompt = torch.tensor([[[2, 33, 33], [3, 33, 33], [4, 33, 33]]])
    text = torch.ones((1, 3), dtype=torch.bool)
    empty = torch.zeros_like(text)
    output = model.generate(
        prompt,
        text,
        empty,
        empty,
        max_new_tokens=2,
        gen_aud=True,
        temperature=0,
    )
    generated = output[:, -2:, 1:]
    assert generated.min() >= 0
    assert generated.max() < 33


def test_text_prefixed_depth_model_is_pretraining_only(tmp_path):
    model = build_model(
        tmp_path,
        n_codebooks=2,
        codebook_transformer=True,
        text_prefix=True,
    ).eval()
    prompt = torch.tensor([[[2, 33, 33], [3, 33, 33], [4, 33, 33]]])
    text = torch.ones((1, 3), dtype=torch.bool)
    empty = torch.zeros_like(text)
    with pytest.raises(ValueError, match="pretraining-only"):
        model.generate(
            prompt,
            text,
            empty,
            empty,
            max_new_tokens=1,
            gen_aud=True,
            temperature=0,
        )


@pytest.mark.parametrize(
    "temperature,top_k,match",
    [(-1.0, None, "non-negative"), (float("nan"), None, "finite"), (1.0, 0, "positive")],
)
def test_generation_argument_validation(tmp_path, temperature, top_k, match):
    model = build_model(tmp_path).eval()
    prompt = torch.tensor([[[2, 33], [3, 33]]])
    txt = torch.ones((1, 2), dtype=torch.bool)
    empty = torch.zeros_like(txt)
    with pytest.raises(ValueError, match=match):
        model.generate(
            prompt,
            txt,
            empty,
            empty,
            max_new_tokens=1,
            gen_aud=True,
            temperature=temperature,
            top_k=top_k,
        )


def test_codebook_transformer_forward_backward_and_generation(tmp_path):
    model = build_model(
        tmp_path,
        n_codebooks=2,
        attention_residual="none",
        codebook_transformer=True,
    )
    tokens = torch.full((2, 9, 3), 100, dtype=torch.long)
    tokens[..., 1:] = torch.randint(0, 32, (2, 9, 2))
    audio_input = torch.ones((2, 9), dtype=torch.bool)
    empty_input = torch.zeros_like(audio_input)
    audio_pred = torch.ones((2, 8), dtype=torch.bool)
    empty_pred = torch.zeros_like(audio_pred)
    model(
        tokens,
        empty_input,
        empty_input,
        audio_input,
        empty_pred,
        empty_pred,
        audio_pred,
    )
    assert torch.isfinite(model.last_loss)
    model.last_loss.backward()

    model.eval()
    output = model.generate(
        tokens[:1, :4],
        empty_input[:1, :4],
        empty_input[:1, :4],
        audio_input[:1, :4],
        max_new_tokens=3,
        gen_aud=True,
        temperature=0,
    )
    assert output.shape == (1, 7, 3)
    assert output[:, 4:, 1:].min() >= 0
    assert output[:, 4:, 1:].max() < 33


def test_text_prefixed_depth_transformer_uses_aligned_targets(tmp_path):
    model = build_model(
        tmp_path,
        n_codebooks=2,
        attention_residual="none",
        codebook_transformer=True,
        text_prefix=True,
    ).eval()
    tokens = torch.full((1, 5, 3), 100, dtype=torch.long)
    tokens[..., 1:] = torch.randint(0, 32, (1, 5, 2))
    tokens[0, 1, 0] = 7
    text_in = tokens[..., 0] != 100
    audio_in = torch.ones((1, 5), dtype=torch.bool)
    empty_in = torch.zeros_like(audio_in)
    pred = torch.zeros((1, 4), dtype=torch.bool)
    pred[:, 0] = True
    empty_pred = torch.zeros_like(pred)

    text_logits, _, audio_logits = model(
        tokens,
        text_in,
        empty_in,
        audio_in,
        pred,
        empty_pred,
        pred,
        compute_loss=False,
    )
    assert torch.isfinite(text_logits[:, -2:]).all()

    changed_text = tokens.clone()
    changed_text[0, 1, 0] = 8
    _, _, changed_text_logits = model(
        changed_text,
        changed_text[..., 0] != 100,
        empty_in,
        audio_in,
        pred,
        empty_pred,
        pred,
        compute_loss=False,
    )
    assert not torch.allclose(audio_logits, changed_text_logits)

    changed_audio = tokens.clone()
    changed_audio[0, 1, 1] = (changed_audio[0, 1, 1] + 1) % 32
    _, _, changed_audio_logits = model(
        changed_audio,
        text_in,
        empty_in,
        audio_in,
        pred,
        empty_pred,
        pred,
        compute_loss=False,
    )
    torch.testing.assert_close(audio_logits[:, 0], changed_audio_logits[:, 0])
    assert not torch.allclose(audio_logits[:, 1], changed_audio_logits[:, 1])


def test_moshi_config_uses_small_projected_depth_transformer(tmp_path):
    document = json.loads(
        (Path(__file__).parents[1] / "configs/moshi-speech/2b.json").read_text()
    )
    config = document["model"]
    assert document["train_data"]["aud_tokens"] == "mimi"
    assert document["train_data"]["audtxt_mode"] == "aligned"
    assert document["training"]["batch_size"] == 5
    assert document["training"]["gradient_accumulation_steps"] == 96
    assert config["aud_inadapter_dim"] == 2048
    assert config["aud_codebook_transformer_dim"] == 256
    assert config["aud_codebook_transformer_layers"] == 2
    assert config["aud_delay_pattern"] == [0, 2, 2, 2, 2, 2, 2, 2]
    assert config["aud_codebook_weights"] == [100, 1, 1, 1, 1, 1, 1, 1]
    assert config["tie_aud_embeddings"] is False

    tiny = build_model(
        tmp_path,
        n_codebooks=2,
        codebook_transformer=True,
        depth_dim=16,
    )
    projection = tiny.aud_expression.in2codebookt_proj
    assert projection.in_features == 32
    assert projection.out_features == 16
    assert tiny.aud_expression.unembed[0].in_features == 16


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_attention_residual_modes(tmp_path, mode):
    model = build_model(tmp_path, attention_residual=mode)
    tokens = torch.full((1, 7, 2), 100, dtype=torch.long)
    tokens[..., 1] = torch.randint(0, 32, (1, 7))
    audio_input = torch.ones((1, 7), dtype=torch.bool)
    empty_input = torch.zeros_like(audio_input)
    audio_pred = torch.ones((1, 6), dtype=torch.bool)
    empty_pred = torch.zeros_like(audio_pred)
    model(
        tokens,
        empty_input,
        empty_input,
        audio_input,
        empty_pred,
        empty_pred,
        audio_pred,
    )
    assert torch.isfinite(model.last_loss)
