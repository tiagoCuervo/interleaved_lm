import sys
import types

import numpy as np
import pytest
import torch

from interleaved_speech_data import PackedShard, PreparedSample, ShardWriter, validate_shard
from interleaved_speech_data.backends import (
    MimiTokenizer,
    align_words,
    align_words_to_mimi,
    collapse_codes_with_mapping,
)
from interleaved_speech_data.schema import ManifestRecord, manifest_fingerprint, read_manifest, write_manifest
from interleaved_speech_data.sources import fairseq_tsv_records, librispeech_records, make_shard_plan
from interleaved_speech_data.tinystories import _kokoro_synthesize


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(text[0])]}


def test_mimi_tokenizer_emits_eight_streams_and_boundary(tmp_path, monkeypatch):
    weights = tmp_path / "mimi.safetensors"
    weights.write_bytes(b"mimi")

    class Model:
        def encode(self, waveform):
            assert waveform.shape == (1, 1, 2400)
            return torch.arange(32).reshape(1, 8, 4)

    loaders = types.SimpleNamespace(
        SAMPLE_RATE=24000,
        FRAME_RATE=12.5,
        get_mimi=lambda path, device, num_codebooks: Model(),
    )
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: str(weights)
    models = types.ModuleType("moshi.models")
    models.loaders = loaders
    moshi = types.ModuleType("moshi")
    moshi.models = models
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "moshi", moshi)
    monkeypatch.setitem(sys.modules, "moshi.models", models)

    tokenizer = MimiTokenizer("kyutai/moshi", "revision", device="cpu")
    codes, frame_to_code = tokenizer.encode_with_mapping(
        np.zeros(2400, dtype=np.float32)
    )
    assert codes.shape == (5, 8)
    assert np.all(codes[-1] == 2048)
    assert frame_to_code.tolist() == [0, 1, 2, 3, 4]
    assert tokenizer.metadata["frame_rate"] == 12.5


def test_collapsed_alignment_uses_the_uncollapsed_timeline():
    collapsed, frame_to_code = collapse_codes_with_mapping(
        np.array([1, 1, 1, 2, 3, 3, 3, 4, 5, 5], dtype=np.int16), eos_token=6
    )
    assert collapsed[:, 0].tolist() == [1, 2, 3, 4, 5, 6]
    assert frame_to_code.tolist() == [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 5]
    tokens, durations = align_words(
        [
            {"text": "a", "start": 0.0, "end": 0.4},
            {"text": "b", "start": 0.4, "end": 0.8},
        ],
        _Tokenizer(),
        len(collapsed),
        audio_seconds=1.0,
        pad_id=100,
        end_of_word_id=101,
        frame_to_code=frame_to_code,
    )
    assert np.repeat(tokens, durations).tolist() == [
        ord("a"),
        101,
        ord("b"),
        101,
        100,
        100,
    ]


def test_alignment_uses_adjacent_silence_for_short_words():
    tokens, durations = align_words(
        [{"text": "a", "start": 0.0, "end": 0.1}],
        _Tokenizer(),
        4,
        audio_seconds=1.0,
        pad_id=100,
        end_of_word_id=101,
    )
    assert np.repeat(tokens, durations).tolist() == [ord("a"), 101, 100, 100]


def test_mimi_alignment_places_epad_before_the_next_word():
    tokens, durations = align_words_to_mimi(
        [
            {"text": "a", "start": 0.0, "end": 0.1},
            {"text": "b", "start": 0.5, "end": 0.6},
        ],
        _Tokenizer(),
        9,
        audio_seconds=1.0,
        pad_id=100,
        end_of_word_id=101,
    )
    assert np.repeat(tokens, durations).tolist() == [
        ord("a"),
        100,
        100,
        101,
        ord("b"),
        100,
        100,
        101,
        100,
    ]


def test_kokoro_timestamps_drop_punctuation_only_tokens():
    class Token:
        def __init__(self, text, start, end):
            self.text, self.start_ts, self.end_ts = text, start, end

    class Result:
        audio = torch.zeros(2400)
        tokens = [Token("hello", 0.0, 0.08), Token("!", 0.08, 0.1)]

    waveform, words = _kokoro_synthesize(lambda *args, **kwargs: [Result()], "hello!", "voice")
    assert waveform.shape == (2400,)
    assert words == [{"text": "hello", "start": 0.0, "end": 0.08}]


def test_manifest_index_and_deterministic_plan(tmp_path):
    records = [
        ManifestRecord(id=f"d:train:{i}", dataset="d", split="train", audio=f"{i}.wav", duration=i + 1)
        for i in range(9)
    ]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    assert [record.id for record in read_manifest(manifest, [8, 0, 3])] == ["d:train:8", "d:train:0", "d:train:3"]
    first = make_shard_plan(manifest, 3, tmp_path / "plan-a.json")
    second = make_shard_plan(manifest, 3, tmp_path / "plan-b.json")
    assert first == second
    assert sorted(i for shard in first["shards"] for i in shard["record_indices"]) == list(range(9))


def test_plan_rejects_empty_shards_and_detects_manifest_changes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [ManifestRecord(id="d:train:0", dataset="d", split="train", audio="0.wav")],
        manifest,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        make_shard_plan(manifest, 2, tmp_path / "too-many.json")
    plan = make_shard_plan(manifest, 1, tmp_path / "plan.json")
    assert plan["manifest_sha256"]
    manifest.write_text(manifest.read_text() + "\n")
    assert plan["manifest_sha256"] != manifest_fingerprint(manifest)


def test_units_and_aligned_shards(tmp_path):
    for profile in ("units-only", "aligned"):
        root = tmp_path / profile
        with ShardWriter(root, "shard-00000", profile=profile, n_codebooks=2, preprocessing={"test": True}) as writer:
            for index in range(3):
                codes = np.arange(20, dtype=np.int16).reshape(10, 2)
                writer.write(
                    PreparedSample(
                        id=str(index),
                        codes=codes,
                        text_tokens=np.array([1, 2, 3], np.int32) if profile == "aligned" else None,
                        token_durations=np.array([3, 3, 4], np.int32) if profile == "aligned" else None,
                    )
                )
        meta = validate_shard(root / "shard-00000")
        assert meta["num_samples"] == 3
        shard = PackedShard(root / "shard-00000")
        assert shard.sample(1)["audio"].shape == (10, 2)
        if profile == "aligned":
            assert np.repeat(shard.sample(1)["text"], shard.sample(1)["durations"]).shape == (10,)


def test_completed_shard_is_idempotent(tmp_path):
    with ShardWriter(tmp_path, "part", profile="units-only", n_codebooks=1, preprocessing={}) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones((4, 1), np.int16)))
    with ShardWriter(tmp_path, "part", profile="units-only", n_codebooks=1, preprocessing={}) as writer:
        assert writer.already_complete


def test_completed_shard_rejects_different_preprocessing(tmp_path):
    with ShardWriter(
        tmp_path,
        "part",
        profile="units-only",
        n_codebooks=1,
        preprocessing={"revision": "one"},
    ) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones((4, 1), np.int16)))
    with pytest.raises(ValueError, match="different parameters"):
        with ShardWriter(
            tmp_path,
            "part",
            profile="units-only",
            n_codebooks=1,
            preprocessing={"revision": "two"},
        ):
            pass


def test_writer_rejects_nonpositive_codebook_count(tmp_path):
    with pytest.raises(ValueError, match="n_codebooks must be positive"):
        ShardWriter(
            tmp_path,
            "part",
            profile="units-only",
            n_codebooks=0,
            preprocessing={},
        )


def test_rejected_sample_does_not_corrupt_shard(tmp_path):
    with ShardWriter(tmp_path, "part", profile="aligned", n_codebooks=1, preprocessing={}) as writer:
        with pytest.raises(ValueError, match="does not cover"):
            writer.write(
                PreparedSample(
                    id="bad",
                    codes=np.ones((4, 1), np.int16),
                    text_tokens=np.array([1, 2], np.int32),
                    token_durations=np.array([1, 1], np.int32),
                )
            )
        writer.write(
            PreparedSample(
                id="good",
                codes=np.ones((4, 1), np.int16),
                text_tokens=np.array([1, 2], np.int32),
                token_durations=np.array([2, 2], np.int32),
            )
        )
    meta = validate_shard(tmp_path / "part")
    assert meta["num_samples"] == 1
    assert meta["num_audio_steps"] == 4


def test_interrupted_shard_is_never_published(tmp_path):
    with pytest.raises(RuntimeError):
        with ShardWriter(tmp_path, "part", profile="units-only", n_codebooks=1, preprocessing={}) as writer:
            writer.write(PreparedSample(id="a", codes=np.ones((4, 1), np.int16)))
            raise RuntimeError("interrupted")
    assert not (tmp_path / "part").exists()


def test_empty_shard_keeps_machine_readable_skip_diagnostics(tmp_path):
    with pytest.raises(ValueError, match="diagnostics"):
        with ShardWriter(
            tmp_path,
            "part",
            profile="units-only",
            n_codebooks=1,
            preprocessing={"num_assigned_records": 1},
        ) as writer:
            writer.skip("sample", "ValueError", "bad input")
    assert not (tmp_path / "part").exists()
    assert '"id": "sample"' in (tmp_path / "part.failed.skips.jsonl").read_text()
    assert '"num_skipped": 1' in (tmp_path / "part.failed.json").read_text()


def test_checksum_mismatch_is_rejected(tmp_path):
    with ShardWriter(tmp_path, "part", profile="units-only", n_codebooks=1, preprocessing={}) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones((4, 1), np.int16)))
    with (tmp_path / "part" / "audio.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="audio size"):
        validate_shard(tmp_path / "part")


def test_metadata_cannot_override_schema_fields(tmp_path):
    with pytest.raises(ValueError, match="reserved fields"):
        with ShardWriter(
            tmp_path,
            "part",
            profile="units-only",
            n_codebooks=1,
            preprocessing={},
        ) as writer:
            writer.write(
                PreparedSample(
                    id="real-id",
                    codes=np.ones((4, 1), np.int16),
                    metadata={"id": "spoofed-id"},
                )
            )


def test_sample_metadata_count_is_validated(tmp_path):
    with ShardWriter(
        tmp_path,
        "part",
        profile="units-only",
        n_codebooks=1,
        preprocessing={"vocab_size": 2},
    ) as writer:
        writer.write(PreparedSample(id="a", codes=np.ones((4, 1), np.int16)))
    (tmp_path / "part" / "samples.jsonl").write_text("")
    with pytest.raises(ValueError, match="metadata count"):
        validate_shard(tmp_path / "part", verify_checksums=False)


def test_librispeech_adapter_indexes_audio_once_and_rejects_duplicates(tmp_path):
    chapter = tmp_path / "1" / "2"
    chapter.mkdir(parents=True)
    (chapter / "1-2.trans.txt").write_text("1-2-3 HELLO\n1-2-4 WORLD\n")
    (chapter / "1-2-3.flac").touch()
    (chapter / "1-2-4.wav").touch()
    records = list(librispeech_records(tmp_path, "librispeech", "train"))
    assert [record.id for record in records] == [
        "librispeech:train:1-2-3",
        "librispeech:train:1-2-4",
    ]
    (chapter / "1-2-3.wav").touch()
    with pytest.raises(ValueError, match="duplicate LibriSpeech audio"):
        list(librispeech_records(tmp_path, "librispeech", "train"))


def test_fairseq_adapter_resolves_relative_root_from_manifest(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = manifest_dir / "train.tsv"
    manifest.write_text("../audio\nexample.flac\t32000\n")
    record = list(fairseq_tsv_records(manifest, "d", "train"))[0]
    assert record.audio == str((audio / "example.flac").resolve())
    assert record.duration == 2.0
