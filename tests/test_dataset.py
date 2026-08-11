import torch
import pytest

from interleaved_lm.dataset import DatasetArgs, MultimodalPretrainDataset
from tests.fixtures import create_toy_dataset


def test_canonical_aligned_speech_is_consumed_directly(tmp_path):
    create_toy_dataset(tmp_path / "toy" / "hubert", profile="aligned")
    config = DatasetArgs(
        data_root=str(tmp_path),
        audtxt_datasets=["toy"],
        splits=["train"],
        p_strategies=[0, 0, 0, 0, 1],
        txt_tokens="smollm",
        aud_tokens="hubert",
        block_size=32,
    )
    dataset = MultimodalPretrainDataset(config)
    assert dataset.txt_pad_token == 100
    assert dataset.txt_epad_token == 101
    assert dataset.txt_swt_token == 102
    assert dataset.txt_vocabsize == 103
    assert len({dataset.txt_pad_token, dataset.txt_epad_token, dataset.txt_swt_token}) == 3
    tokens, txt_in, img_in, aud_in, txt_pred, img_pred, aud_pred = next(iter(dataset))
    assert tokens.shape == (33, 2)
    assert txt_in.any() or aud_in.any()
    assert not img_in.any()
    assert torch.all(~txt_pred | txt_in)
    assert torch.all(~aud_pred | aud_in)


def test_moshi_style_aligned_streams_are_synchronized(tmp_path):
    create_toy_dataset(
        tmp_path / "toy" / "multicodebook",
        profile="aligned",
        n_codebooks=2,
    )
    dataset = MultimodalPretrainDataset(
        DatasetArgs(
            data_root=str(tmp_path),
            audtxt_datasets=["toy"],
            splits=["train"],
            p_strategies=[0, 0, 0, 0, 1],
            txt_tokens="smollm",
            aud_tokens="multicodebook",
            audtxt_mode="aligned",
            block_size=32,
        )
    )
    tokens, txt_in, img_in, aud_in, txt_pred, img_pred, aud_pred = next(iter(dataset))
    assert tokens.shape == (33, 3)
    assert dataset.txt_swt_token is None
    assert dataset.txt_vocabsize == 102
    assert torch.all(aud_in)
    assert torch.all(txt_pred)
    assert torch.all(aud_pred)
    assert not img_in.any()
    assert not img_pred.any()
    assert torch.equal(txt_in, tokens[:, 0] != dataset.txt_pad_token)


def test_dataset_configuration_rejects_ambiguous_weights(tmp_path):
    with pytest.raises(ValueError, match="must contain"):
        MultimodalPretrainDataset(
            DatasetArgs(data_root=str(tmp_path), p_strategies=[1, 0])
        )
    with pytest.raises(ValueError, match="must match"):
        MultimodalPretrainDataset(
            DatasetArgs(
                data_root=str(tmp_path),
                txt_datasets=["a", "b"],
                txt_datasets_probs=[1.0],
                splits=["train"],
                p_strategies=[1, 0, 0, 0, 0],
            )
        )


def test_unversioned_bins_are_not_loaded(tmp_path):
    root = tmp_path / "toy" / "hubert"
    root.mkdir(parents=True)
    (root / "meta_aud.json").write_text(
        '{"vocab_size":34,"pad_id":33,"eos_id":32,"n_q":1}'
    )
    (root / "train.bin").write_bytes(b"unversioned")
    (root / "train.len").write_bytes(b"unversioned")
    with pytest.raises(ValueError, match="no aud data"):
        MultimodalPretrainDataset(
            DatasetArgs(
                data_root=str(tmp_path),
                aud_datasets=["toy"],
                splits=["train"],
                p_strategies=[0, 0, 1, 0, 0],
                aud_tokens="hubert",
                block_size=4,
            )
        )
