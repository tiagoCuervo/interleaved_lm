from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .backends import align_words, align_words_to_mimi, build_speech_tokenizer
from .packing import PreparedSample, ShardWriter
from .schema import ManifestRecord, manifest_fingerprint, read_manifest, sha256_file


def tinystories_manifest_records(
    dataset_id: str,
    dataset_revision: str,
    split: str,
    *,
    max_samples: int | None = None,
) -> Iterable[ManifestRecord]:
    """Read TinyStories once and emit indexed synthesis records for balanced planning."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, revision=dataset_revision)
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for index in range(total):
        text = str(dataset[index]["text"]).strip()
        yield ManifestRecord(
            id=f"tinystories:{split}:{index:08d}",
            dataset="tinystories",
            split=split,
            audio=None,
            transcript=text,
            source_revision=dataset_revision,
            extra={"source_dataset": dataset_id, "source_index": index},
        )


def _kokoro_synthesize(pipeline, text: str, voice: str, sample_rate: int = 24000):
    chunks: list[np.ndarray] = []
    words: list[dict] = []
    time_offset = 0.0
    for result in pipeline(text, voice=voice):
        if result.audio is None:
            raise ValueError("Kokoro returned a segment without audio")
        audio = result.audio.detach().cpu().float().reshape(-1).numpy()
        chunks.append(audio)
        for token in result.tokens or []:
            word = str(getattr(token, "text", "") or "").strip()
            start = getattr(token, "start_ts", None)
            end = getattr(token, "end_ts", None)
            if any(character.isalnum() for character in word) and start is not None and end is not None:
                words.append({"text": word, "start": float(start) + time_offset, "end": float(end) + time_offset})
        time_offset += len(audio) / sample_rate
    if not chunks:
        raise ValueError("Kokoro returned no audio")
    return np.concatenate(chunks).astype(np.float32), words


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform
    try:
        from torchaudio.functional import resample
    except ImportError as exc:
        raise ImportError("Kokoro preparation requires torchaudio for resampling") from exc
    return resample(
        torch.from_numpy(waveform).view(1, -1), source_rate, target_rate
    ).view(-1).numpy()


def synthesize_tinystories(
    output_dir: str,
    *,
    profile: str,
    shard_index: int,
    manifest: str,
    plan_path: str,
    speech_tokenizer: str,
    hubert_model: str | None = None,
    hubert_revision: str | None = None,
    kmeans: str | list[str] | None = None,
    hubert_layer: int = 11,
    hubert_normalize: bool | None = None,
    mimi_repo: str | None = None,
    mimi_revision: str | None = None,
    mimi_filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
    device: str = "cuda",
    dataset_id: str = "roneneldan/TinyStories",
    dataset_revision: str,
    kokoro_id: str = "hexgrad/Kokoro-82M",
    kokoro_revision: str,
    voice: str = "af_heart",
    language: str = "a",
    text_tokenizer: str | None = None,
    text_tokenizer_revision: str | None = None,
    split: str = "train",
    keep_audio: bool = False,
    seed: int = 1337,
):
    plan = json.loads(Path(plan_path).read_text())
    if plan["manifest"] != str(Path(manifest).resolve()):
        raise ValueError("shard plan was created for a different manifest")
    if plan.get("manifest_sha256") != manifest_fingerprint(manifest):
        raise ValueError("manifest contents changed after the shard plan was created")
    try:
        shard = plan["shards"][shard_index]
    except IndexError as exc:
        raise ValueError("shard_index is outside the plan") from exc
    if shard["index"] != shard_index:
        raise ValueError("malformed shard plan")
    assignment_sha256 = hashlib.sha256(
        json.dumps(shard["record_indices"], separators=(",", ":")).encode()
    ).hexdigest()
    records = list(read_manifest(manifest, shard["record_indices"]))
    if any(record.source_revision != dataset_revision for record in records):
        raise ValueError("manifest and requested TinyStories revisions differ")
    if any(record.split != split for record in records):
        raise ValueError("manifest and requested TinyStories splits differ")
    from huggingface_hub import snapshot_download
    from kokoro import KModel, KPipeline

    model_path = snapshot_download(kokoro_id, revision=kokoro_revision)
    model_file = Path(model_path) / "kokoro-v1_0.pth"
    kokoro_model = KModel(
        repo_id=kokoro_id,
        config=str(Path(model_path) / "config.json"),
        model=str(model_file),
    ).to(device).eval()
    pipeline = KPipeline(lang_code=language, repo_id=kokoro_id, model=kokoro_model)
    voice_path = str(Path(model_path) / "voices" / f"{voice}.pt")
    unit_tokenizer = build_speech_tokenizer(
        speech_tokenizer,
        device=device,
        hubert_model=hubert_model,
        hubert_revision=hubert_revision,
        kmeans=kmeans,
        hubert_layer=hubert_layer,
        hubert_normalize=hubert_normalize,
        mimi_repo=mimi_repo,
        mimi_revision=mimi_revision,
        mimi_filename=mimi_filename,
    )

    tokenizer = None
    pad_id = end_of_word_id = None
    if profile == "aligned":
        if not text_tokenizer:
            raise ValueError("aligned TinyStories preparation requires --text-tokenizer")
        from transformers import AutoTokenizer

        if not text_tokenizer_revision:
            raise ValueError("aligned TinyStories preparation requires --text-tokenizer-revision")
        tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer, revision=text_tokenizer_revision
        )
        pad_id, end_of_word_id = tokenizer.vocab_size, tokenizer.vocab_size + 1
    preprocessing = {
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "manifest_sha256": manifest_fingerprint(manifest),
        "plan_format_version": plan.get("format_version"),
        "plan_num_shards": len(plan["shards"]),
        "assignment_sha256": assignment_sha256,
        "kokoro": kokoro_id,
        "kokoro_revision": kokoro_revision,
        "voice": voice,
        "language": language,
        "synthesis_sample_rate": 24000,
        "unit_sample_rate": unit_tokenizer.sample_rate,
        **unit_tokenizer.metadata,
        "vocab_size": unit_tokenizer.pad_token + 1,
        "eos_token": unit_tokenizer.eos_token,
        "pad_token": unit_tokenizer.pad_token,
        "text_tokenizer": text_tokenizer,
        "text_tokenizer_revision": text_tokenizer_revision,
        "num_assigned_records": len(records),
    }
    if speech_tokenizer == "hubert":
        paths = [kmeans] if isinstance(kmeans, str) else kmeans
        preprocessing.update(
            hubert_model=hubert_model,
            hubert_revision=hubert_revision,
            hubert_layer=hubert_layer,
            hubert_normalize=unit_tokenizer.normalize,
            kmeans=[
                {"name": Path(path).name, "sha256": sha256_file(path)}
                for path in paths
            ],
        )
    if tokenizer is not None:
        preprocessing.update(
            text_base_vocab_size=tokenizer.vocab_size,
            text_eos_id=tokenizer.eos_token_id,
            text_pad_id=pad_id,
            text_end_of_word_id=end_of_word_id,
            text_special_vocab_size=tokenizer.vocab_size + 2,
        )
    name = f"shard-{shard_index:05d}"
    with ShardWriter(
        output_dir,
        name,
        profile=profile,
        n_codebooks=unit_tokenizer.n_codebooks,
        preprocessing=preprocessing,
        keep_audio=keep_audio,
    ) as writer:
        if writer.already_complete:
            return writer.final / "meta.json"
        for record in records:
            index = int(record.extra["source_index"])
            sample_id = f"tinystories-kokoro:{record.split}:{index:08d}"
            try:
                sample_seed = (seed + 1_000_003 * index) % 2**32
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                text = (record.transcript or "").strip()
                if not text:
                    raise ValueError("empty story")
                audio_24k, words = _kokoro_synthesize(pipeline, text, voice_path)
                unit_audio = _resample(
                    audio_24k, 24000, unit_tokenizer.sample_rate
                )
                frame_to_code = None
                if profile == "aligned":
                    codes, frame_to_code = unit_tokenizer.encode_with_mapping(unit_audio)
                else:
                    codes = unit_tokenizer.encode(unit_audio)
                text_ids = durations = None
                if profile == "aligned":
                    if not words:
                        raise ValueError("Kokoro returned no word timestamps")
                    align = (
                        align_words_to_mimi
                        if speech_tokenizer == "mimi"
                        else align_words
                    )
                    kwargs = (
                        {}
                        if speech_tokenizer == "mimi"
                        else {"frame_to_code": frame_to_code}
                    )
                    text_ids, durations = align(
                        words,
                        tokenizer,
                        len(codes),
                        audio_seconds=len(audio_24k) / 24000,
                        pad_id=pad_id,
                        end_of_word_id=end_of_word_id,
                        **kwargs,
                    )
            except Exception as exc:
                writer.skip(sample_id, type(exc).__name__, str(exc))
                continue
            writer.write(
                PreparedSample(
                    id=sample_id,
                    codes=codes,
                    text_tokens=text_ids,
                    token_durations=durations,
                    waveform=unit_audio if keep_audio else None,
                    sample_rate=unit_tokenizer.sample_rate,
                    metadata={
                        "dataset": "tinystories-kokoro",
                        "split": record.split,
                        "source_dataset": record.extra["source_dataset"],
                        "source_revision": record.source_revision,
                        "source_index": index,
                    },
                )
            )
    return writer.final / "meta.json"
