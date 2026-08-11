from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .backends import (
    MMSForcedAligner,
    align_words,
    align_words_to_mimi,
    build_speech_tokenizer,
    load_audio,
)
from .packing import PreparedSample, ShardWriter
from .schema import manifest_fingerprint, read_manifest, sha256_file


def encode_shard(
    manifest: str,
    plan_path: str,
    shard_index: int,
    output_dir: str,
    *,
    profile: str,
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
    text_tokenizer: str | None = None,
    text_tokenizer_revision: str | None = None,
    alignment_source: str = "timestamps",
    keep_audio: bool = False,
) -> dict:
    plan = json.loads(Path(plan_path).read_text())
    if plan["manifest"] != str(Path(manifest).resolve()):
        raise ValueError("shard plan was created for a different manifest")
    if plan.get("manifest_sha256") != manifest_fingerprint(manifest):
        raise ValueError("manifest contents changed after the shard plan was created")
    try:
        shard = plan["shards"][shard_index]
    except IndexError as exc:
        raise ValueError(f"shard index {shard_index} is outside the plan") from exc
    if shard["index"] != shard_index:
        raise ValueError("malformed shard plan")
    assignment_sha256 = hashlib.sha256(
        json.dumps(shard["record_indices"], separators=(",", ":")).encode()
    ).hexdigest()

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
            raise ValueError("aligned preparation requires --text-tokenizer")
        from transformers import AutoTokenizer

        if not text_tokenizer_revision:
            raise ValueError("aligned preparation requires --text-tokenizer-revision")
        tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer, revision=text_tokenizer_revision
        )
        base_vocab = tokenizer.vocab_size
        pad_id, end_of_word_id = base_vocab, base_vocab + 1
    if alignment_source not in {"timestamps", "mms"}:
        raise ValueError("alignment_source must be timestamps or mms")
    forced_aligner = MMSForcedAligner(device) if profile == "aligned" and alignment_source == "mms" else None

    preprocessing = {
        "manifest_sha256": manifest_fingerprint(manifest),
        "plan_format_version": plan.get("format_version"),
        "plan_num_shards": len(plan["shards"]),
        "assignment_sha256": assignment_sha256,
        "num_assigned_records": len(shard["record_indices"]),
        "profile": profile,
        **unit_tokenizer.metadata,
        "vocab_size": unit_tokenizer.pad_token + 1,
        "eos_token": unit_tokenizer.eos_token,
        "pad_token": unit_tokenizer.pad_token,
        "text_tokenizer": text_tokenizer,
        "text_tokenizer_revision": text_tokenizer_revision,
        "alignment_source": alignment_source,
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
            return json.loads((writer.final / "meta.json").read_text())
        for record in read_manifest(manifest, shard["record_indices"]):
            try:
                if record.audio is None:
                    raise ValueError("record has no audio path")
                waveform, sample_rate = load_audio(
                    record.audio,
                    offset=record.offset,
                    duration=record.duration,
                    target_rate=unit_tokenizer.sample_rate,
                )
                frame_to_code = None
                if profile == "aligned":
                    codes, frame_to_code = unit_tokenizer.encode_with_mapping(waveform)
                else:
                    codes = unit_tokenizer.encode(waveform)
                text_ids = durations = None
                if profile == "aligned":
                    words = (
                        forced_aligner.align(waveform, record.transcript or "")
                        if forced_aligner is not None
                        else record.words
                    )
                    if not words:
                        raise ValueError(f"record has no {alignment_source} alignment")
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
                        audio_seconds=len(waveform) / sample_rate,
                        pad_id=pad_id,
                        end_of_word_id=end_of_word_id,
                        **kwargs,
                    )
            except Exception as exc:
                writer.skip(record.id, type(exc).__name__, str(exc))
                continue
            writer.write(
                PreparedSample(
                    id=record.id,
                    codes=codes,
                    text_tokens=text_ids,
                    token_durations=durations,
                    waveform=waveform if keep_audio else None,
                    sample_rate=sample_rate,
                    metadata={
                        **record.extra,
                        "dataset": record.dataset,
                        "split": record.split,
                        "source_revision": record.source_revision,
                        "speaker": record.speaker,
                        "source_audio_name": Path(record.audio).name,
                        "source_offset": record.offset,
                        "source_duration": record.duration,
                    },
                )
            )
    return json.loads((Path(output_dir) / name / "meta.json").read_text())
