from __future__ import annotations

import json
import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schema import SCHEMA_VERSION, sha256_file, validate_shard


@dataclass
class PreparedSample:
    id: str
    codes: np.ndarray
    text_tokens: np.ndarray | None = None
    token_durations: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    waveform: np.ndarray | None = None
    sample_rate: int | None = None


class ShardWriter:
    """Atomic writer for one independently retryable packed-data shard."""

    def __init__(
        self,
        output_dir: str | Path,
        shard_name: str,
        *,
        profile: str,
        n_codebooks: int,
        preprocessing: dict[str, Any],
        keep_audio: bool = False,
    ):
        if profile not in {"units-only", "aligned"}:
            raise ValueError("profile must be units-only or aligned")
        if n_codebooks < 1:
            raise ValueError("n_codebooks must be positive")
        self.output_dir = Path(output_dir)
        self.final = self.output_dir / shard_name
        self.profile = profile
        self.n_codebooks = n_codebooks
        self.preprocessing = preprocessing
        self.keep_audio = keep_audio
        self.tmp = self.output_dir / f".{shard_name}.tmp-{uuid.uuid4().hex}"
        self._files: dict[str, Any] = {}
        self._samples = None
        self._skips = None
        self._skip_count = 0
        self._count = 0
        self._audio_steps = 0
        self._text_tokens = 0
        self._closed = False

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.final.exists():
            meta = validate_shard(self.final)
            if (
                meta.get("profile") != self.profile
                or int(meta.get("n_codebooks", -1)) != self.n_codebooks
                or meta.get("preprocessing") != self.preprocessing
                or bool(meta.get("keep_audio", False)) != self.keep_audio
            ):
                raise ValueError(
                    f"completed shard {self.final} was prepared with different parameters"
                )
            self._closed = True
            return self
        self.tmp.mkdir()
        self._files["audio.bin"] = (self.tmp / "audio.bin").open("wb")
        self._files["audio.len"] = (self.tmp / "audio.len").open("wb")
        if self.profile == "aligned":
            self._files["text.bin"] = (self.tmp / "text.bin").open("wb")
            self._files["text.len"] = (self.tmp / "text.len").open("wb")
            self._files["dur.bin"] = (self.tmp / "dur.bin").open("wb")
        self._samples = (self.tmp / "samples.jsonl").open("w")
        self._skips = (self.tmp / "skips.jsonl").open("w")
        return self

    @property
    def already_complete(self) -> bool:
        return self._closed

    def write(self, sample: PreparedSample) -> None:
        if self._closed:
            return
        codes = np.asarray(sample.codes)
        if codes.ndim == 1:
            codes = codes[:, None]
        if codes.ndim != 2 or codes.shape[1] != self.n_codebooks or not len(codes):
            raise ValueError(f"invalid code shape for {sample.id}: {codes.shape}")
        if codes.min() < 0 or codes.max() > np.iinfo(np.int16).max:
            raise ValueError(f"codes outside int16 range for {sample.id}")

        reserved = {"id", "audio_steps", "text_tokens", "prepared_audio"}
        overlap = reserved.intersection(sample.metadata)
        if overlap:
            raise ValueError(
                f"metadata for {sample.id} uses reserved fields: {sorted(overlap)}"
            )
        row = {**sample.metadata, "id": sample.id, "audio_steps": len(codes)}
        text = durations = None
        if self.profile == "aligned":
            if sample.text_tokens is None or sample.token_durations is None:
                raise ValueError(f"aligned profile requires text and durations for {sample.id}")
            text = np.asarray(sample.text_tokens, dtype=np.int32).reshape(-1)
            durations = np.asarray(sample.token_durations, dtype=np.int32).reshape(-1)
            if not len(text) or len(text) != len(durations) or np.any(durations <= 0):
                raise ValueError(f"invalid alignment for {sample.id}")
            if int(durations.sum()) != len(codes):
                raise ValueError(f"alignment does not cover all audio steps for {sample.id}")
            row["text_tokens"] = len(text)

        if self.keep_audio and sample.waveform is not None:
            try:
                import soundfile as sf
            except ImportError as exc:
                raise ImportError("soundfile is required with --keep-audio") from exc
            audio_dir = self.tmp / "waveforms"
            audio_dir.mkdir(exist_ok=True)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.id)
            suffix = hashlib.sha1(sample.id.encode(), usedforsecurity=False).hexdigest()[:10]
            relpath = Path("waveforms") / f"{safe_id}-{suffix}.flac"
            sf.write(self.tmp / relpath, sample.waveform, sample.sample_rate or 16000)
            row["prepared_audio"] = str(relpath)

        codes.astype(np.int16, copy=False).tofile(self._files["audio.bin"])
        np.asarray([len(codes)], dtype=np.int64).tofile(self._files["audio.len"])
        if text is not None:
            text.tofile(self._files["text.bin"])
            np.asarray([len(text)], dtype=np.int64).tofile(self._files["text.len"])
            durations.tofile(self._files["dur.bin"])

        self._samples.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._count += 1
        self._audio_steps += len(codes)
        self._text_tokens += 0 if text is None else len(text)

    def skip(self, sample_id: str, reason: str, detail: str = "") -> None:
        if self._closed:
            return
        self._skips.write(json.dumps({"id": sample_id, "reason": reason, "detail": detail}, sort_keys=True) + "\n")
        self._skip_count += 1

    def __exit__(self, exc_type, exc, traceback):
        if self._closed:
            return False
        for stream in self._files.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        self._samples.flush()
        os.fsync(self._samples.fileno())
        self._samples.close()
        self._skips.flush()
        os.fsync(self._skips.fileno())
        self._skips.close()
        if exc_type is not None:
            return False
        if self._count == 0:
            failure_skips = self.output_dir / f"{self.final.name}.failed.skips.jsonl"
            failure_meta = self.output_dir / f"{self.final.name}.failed.json"
            (self.tmp / "skips.jsonl").replace(failure_skips)
            failure_meta.write_text(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "profile": self.profile,
                "num_assigned_records": self.preprocessing.get("num_assigned_records"),
                "num_skipped": self._skip_count,
                "skips_sha256": sha256_file(failure_skips),
            }, indent=2, sort_keys=True) + "\n")
            raise ValueError(
                f"refusing to publish an empty shard; diagnostics: {failure_meta}"
            )

        files = ["audio.bin", "audio.len", "samples.jsonl", "skips.jsonl"]
        if self.profile == "aligned":
            files += ["text.bin", "text.len", "dur.bin"]
        files += [
            str(path.relative_to(self.tmp))
            for path in sorted((self.tmp / "waveforms").glob("*.flac"))
        ] if (self.tmp / "waveforms").exists() else []
        meta = {
            "schema_version": SCHEMA_VERSION,
            "profile": self.profile,
            "n_codebooks": self.n_codebooks,
            "num_samples": self._count,
            "num_audio_steps": self._audio_steps,
            "num_text_tokens": self._text_tokens,
            "num_skipped": self._skip_count,
            "keep_audio": self.keep_audio,
            "preprocessing": self.preprocessing,
            "checksums": {name: sha256_file(self.tmp / name) for name in files},
        }
        (self.tmp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        validate_shard(self.tmp)
        self.tmp.replace(self.final)
        for failed in (
            self.output_dir / f"{self.final.name}.failed.json",
            self.output_dir / f"{self.final.name}.failed.skips.jsonl",
        ):
            failed.unlink(missing_ok=True)
        self._closed = True
        return False


class PackedShard:
    """Memory-mapped reader for one canonical speech shard."""

    def __init__(self, path: str | Path, verify_checksums: bool = False):
        self.path = Path(path)
        self.meta = validate_shard(self.path, verify_checksums=verify_checksums)
        self.lengths = np.memmap(self.path / "audio.len", dtype=np.int64, mode="r")
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths, dtype=np.int64)))
        self.codes = np.memmap(
            self.path / "audio.bin",
            dtype=np.int16,
            mode="r",
            shape=(int(self.offsets[-1]), int(self.meta["n_codebooks"])),
        )
        self.text_lengths = self.text_offsets = self.text = self.durations = None
        if self.meta["profile"] == "aligned":
            self.text_lengths = np.memmap(self.path / "text.len", dtype=np.int64, mode="r")
            self.text_offsets = np.concatenate(([0], np.cumsum(self.text_lengths, dtype=np.int64)))
            self.text = np.memmap(self.path / "text.bin", dtype=np.int32, mode="r")
            self.durations = np.memmap(self.path / "dur.bin", dtype=np.int32, mode="r")

    def __len__(self) -> int:
        return len(self.lengths)

    def sample(self, index: int) -> dict[str, np.ndarray]:
        lo, hi = int(self.offsets[index]), int(self.offsets[index + 1])
        result = {"audio": np.asarray(self.codes[lo:hi])}
        if self.text is not None:
            tlo, thi = int(self.text_offsets[index]), int(self.text_offsets[index + 1])
            result["text"] = np.asarray(self.text[tlo:thi])
            result["durations"] = np.asarray(self.durations[tlo:thi])
        return result
