"""Prepare deterministic text-only shards for interleaved pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_root_metadata(root: Path, alias: str, entry: dict) -> None:
    path = root / "meta_txt.json"
    payload = {alias: entry}
    temporary = root / f".meta_txt.json.{uuid.uuid4().hex}.tmp"
    root.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    except FileExistsError:
        current = json.loads(path.read_text())
        if current.get(alias) != entry:
            raise ValueError(f"metadata conflict at {path}")
    finally:
        temporary.unlink(missing_ok=True)


def validate(path: str | Path) -> dict:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("modality") != "text":
        raise ValueError(f"unsupported text shard: {path}")
    lengths = np.fromfile(path / "text.len", dtype=np.int64)
    if len(lengths) != meta["num_samples"] or np.any(lengths <= 0):
        raise ValueError(f"invalid text lengths in {path}")
    if int(lengths.sum()) != meta["num_tokens"]:
        raise ValueError(f"text token count mismatch in {path}")
    if (path / "text.bin").stat().st_size != meta["num_tokens"] * np.dtype(np.int32).itemsize:
        raise ValueError(f"text data size mismatch in {path}")
    for name, expected in meta["checksums"].items():
        if _sha256(path / name) != expected:
            raise ValueError(f"checksum mismatch for {path / name}")
    return meta


def prepare(args) -> Path:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    root = Path(args.output)
    final = root / args.split / f"shard-{args.shard_index:05d}"
    request = {
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "name": args.name,
        "split": args.split,
        "text_column": args.text_column,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_alias": args.tokenizer_alias,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_samples": args.max_samples,
    }
    if final.exists():
        meta = validate(final)
        if meta.get("request") != request:
            raise ValueError(f"completed shard {final} was prepared with different parameters")
        return final
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id, revision=args.tokenizer_revision, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("the text tokenizer must define an EOS token")
    dataset = load_dataset(args.dataset, args.name, split=args.split, revision=args.dataset_revision)
    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards

    temporary = root / args.split / f".shard-{args.shard_index:05d}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    streams = {
        "text.bin": (temporary / "text.bin").open("wb"),
        "text.len": (temporary / "text.len").open("wb"),
        "samples.jsonl": (temporary / "samples.jsonl").open("w"),
        "skips.jsonl": (temporary / "skips.jsonl").open("w"),
    }
    count = tokens = skipped = 0
    try:
        for source_index in range(start, end):
            sample_id = f"{args.dataset}:{args.split}:{source_index}"
            try:
                text = str(dataset[source_index][args.text_column]).strip()
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                ids = np.asarray(ids + [tokenizer.eos_token_id], dtype=np.int32)
                if len(ids) < 2:
                    raise ValueError("empty tokenized document")
            except Exception as exc:
                streams["skips.jsonl"].write(
                    json.dumps({"id": sample_id, "reason": type(exc).__name__, "detail": str(exc)}) + "\n"
                )
                skipped += 1
                continue
            ids.tofile(streams["text.bin"])
            np.asarray([len(ids)], dtype=np.int64).tofile(streams["text.len"])
            streams["samples.jsonl"].write(
                json.dumps({"id": sample_id, "source_index": source_index, "tokens": len(ids)}) + "\n"
            )
            count += 1
            tokens += len(ids)
    finally:
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    if not count:
        raise ValueError("refusing to publish an empty text shard")
    files = list(streams)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "modality": "text",
        "num_samples": count,
        "num_tokens": tokens,
        "num_skipped": skipped,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "source_range": [start, end],
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_alias": args.tokenizer_alias,
        "base_vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_token_id,
        "special_vocab_size": tokenizer.vocab_size + 1,
        "request": request,
        "checksums": {name: _sha256(temporary / name) for name in files},
    }
    (temporary / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    validate(temporary)
    entry = {
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_alias": args.tokenizer_alias,
        "base_vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_token_id,
        "special_vocab_size": tokenizer.vocab_size + 1,
    }
    _write_root_metadata(root, args.tokenizer_alias, entry)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tokenizer-alias", default="smollm")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    print(prepare(args))


if __name__ == "__main__":
    main()
