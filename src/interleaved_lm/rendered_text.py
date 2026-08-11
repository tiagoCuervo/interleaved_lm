"""Deterministic, atomic preparation of PIXEL-style rendered text."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
import cairo
import gi
import manimpango
import numpy as np
from datasets import load_dataset
from fontTools import ttLib
from transformers import AutoTokenizer

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo  # noqa: E402


SCHEMA_VERSION = 1
MAX_CAIRO_DIMENSION = 32_767
TOKENIZER_ALIAS = {
    "HuggingFaceTB/SmolLM-135M": "smollm",
    "HuggingFaceTB/SmolLM-360M": "smollm",
    "HuggingFaceTB/SmolLM-1.7B": "smollm",
}


@dataclass
class Encoding:
    pixels: np.ndarray
    words: list[str]
    word_spans: list[tuple[int, int]]
    length: int


class PangoCairoTextRenderer:
    """The bigram-within-word renderer used by the paper experiments."""

    def __init__(
        self,
        font_file: str,
        font_size: int = 8,
        rgb: bool = False,
        dpi: int = 120,
        pad_size: int = 3,
        pixels_per_patch: int = 16,
        max_seq_length: int = 2048,
        **_: object,
    ):
        self.font_file = str(Path(font_file).resolve())
        self.font_size = font_size
        self.rgb = rgb
        self.dpi = dpi
        self.pad_size = pad_size
        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = min(max_seq_length, MAX_CAIRO_DIMENSION // pixels_per_patch)
        self.max_pixels = self.max_seq_length * pixels_per_patch
        manimpango.register_font(self.font_file)
        family = ttLib.TTFont(self.font_file)["name"].getDebugName(1)
        scaled = (dpi / 72) * font_size
        self.font = Pango.font_description_from_string(f"{family} {scaled}px")

    def _surface(self):
        fmt = cairo.FORMAT_RGB24 if self.rgb else cairo.FORMAT_A8
        surface = cairo.ImageSurface(fmt, self.max_pixels, self.pixels_per_patch)
        context = cairo.Context(surface)
        if self.rgb:
            context.set_source_rgb(1, 1, 1)
            context.paint()
            context.set_source_rgb(0, 0, 0)
        return surface, context

    def _width(self, character: str, context: cairo.Context) -> int:
        layout = PangoCairo.create_layout(context)
        layout.set_font_description(self.font)
        layout.set_text(character, -1)
        return layout.get_pixel_size()[0]

    def _draw(self, character: str, x: int, context: cairo.Context) -> int:
        layout = PangoCairo.create_layout(context)
        layout.set_font_description(self.font)
        layout.set_text(character, -1)
        width, height = layout.get_pixel_size()
        context.move_to(x, self.pixels_per_patch / 2 - height / 2 - 2)
        PangoCairo.show_layout(context, layout)
        return width

    def _next_patch(self, pixel: int) -> int:
        return math.ceil(pixel / self.pixels_per_patch) * self.pixels_per_patch

    @staticmethod
    def _is_rtl(text: str) -> bool:
        import string

        text = text.translate(str.maketrans("", "", string.whitespace + string.punctuation + string.digits))
        if not text:
            return False
        characters = (text[0], text[-1], text[len(text) // 2])
        return sum(Pango.unichar_direction(char) == Pango.Direction.RTL for char in characters) >= 2

    @staticmethod
    def _bigrams(word: str, rtl: bool) -> list[str]:
        chunks = [word[index:index + 2] for index in range(0, len(word), 2)]
        return [chunk[::-1] for chunk in chunks][::-1] if rtl else chunks

    def _advance(self, bigram: str, offset: int, context: cairo.Context, *, draw: bool, last: bool) -> int:
        measure = self._draw if draw else self._width
        first = measure(bigram[0], offset, context) if draw else measure(bigram[0], context)
        second = 0
        if len(bigram) == 2:
            second = (
                measure(bigram[1], offset + first, context)
                if draw
                else measure(bigram[1], context)
            )
        return self._next_patch(offset + first + second + (2 if last else 0))

    def __call__(self, text: str, rtl: bool = False, max_length: int | None = None) -> Encoding:
        words = text.strip().split()
        if not words:
            raise ValueError("text is empty")
        rtl = rtl or self._is_rtl(" ".join(words))
        if rtl:
            words = words[::-1]

        limit = self.max_seq_length if max_length is None else min(max_length, self.max_seq_length)
        _, measure_context = self._surface()
        measured_starts = [0]
        offset = 0
        for word in words:
            for bigram in self._bigrams(word, rtl):
                offset = self._advance(bigram, offset, measure_context, draw=False, last=False)
            measured_starts.append(math.ceil(offset / self.pixels_per_patch))
            if offset >= self.max_pixels - self.pixels_per_patch:
                break
        if measured_starts[-1] > limit:
            index = len(measured_starts) - 2
            while index >= 0 and measured_starts[index] > limit:
                index -= 1
            index = max(index - 1, 0)
            words = words[: index + 1]
        if not words:
            raise ValueError("text cannot fit one rendered word")

        surface, context = self._surface()
        offset = 0
        starts = [0]
        for word_index, word in enumerate(words):
            bigrams = self._bigrams(word, rtl)
            for bigram_index, bigram in enumerate(bigrams):
                last = word_index == len(words) - 1 and bigram_index == len(bigrams) - 1
                offset = self._advance(bigram, offset, context, draw=True, last=last)
            starts.append(math.ceil(offset / self.pixels_per_patch))
        eos_patch = min(starts[-1], self.max_seq_length - 1)
        length = eos_patch + 1

        if self.rgb:
            data = np.frombuffer(surface.get_data(), np.uint8).reshape(
                self.pixels_per_patch, self.max_pixels, 4
            )[..., :3][..., ::-1]
            data[:, eos_patch * self.pixels_per_patch:(eos_patch + 1) * self.pixels_per_patch] = 0
            image = data[:, :length * self.pixels_per_patch]
            patches = image.reshape(self.pixels_per_patch, length, self.pixels_per_patch, 3)
            pixels = patches.transpose(1, 0, 2, 3).reshape(length, -1)
        else:
            data = np.invert(
                np.frombuffer(surface.get_data(), np.uint8).reshape(
                    self.pixels_per_patch, self.max_pixels
                )
            )
            data[:, eos_patch * self.pixels_per_patch:(eos_patch + 1) * self.pixels_per_patch] = 0
            image = data[:, :length * self.pixels_per_patch]
            patches = image.reshape(self.pixels_per_patch, length, self.pixels_per_patch)
            pixels = patches.transpose(1, 0, 2).reshape(length, -1)
        return Encoding(
            pixels=pixels.astype(np.int16),
            words=words,
            word_spans=list(zip(starts[:-1], starts[1:])),
            length=length,
        )


def load_renderer_local(renderer_dir: str, **overrides):
    """Load a renderer directory used by both preparation and evaluation."""
    config_path = Path(renderer_dir) / "text_renderer_config.json"
    config = json.loads(config_path.read_text())
    font = Path(config["font_file"])
    if not font.is_absolute():
        config["font_file"] = str(Path(renderer_dir) / font)
    config.update(overrides)
    return PangoCairoTextRenderer(**config), config


def image_to_patch_vectors(enc: Encoding, ppb: int, rgb: bool) -> np.ndarray:
    expected = ppb * ppb * (3 if rgb else 1)
    if enc.pixels.shape != (enc.length, expected):
        raise ValueError(f"unexpected rendered shape {enc.pixels.shape}; expected {(enc.length, expected)}")
    return enc.pixels


def align_text_to_patches(
    enc: Encoding, tokenizer, pad_id: int, epad_id: int, eos_id: int
) -> np.ndarray:
    return align_text(enc, tokenizer, pad_id, epad_id, eos_id)


def align_text(enc: Encoding, tokenizer, pad_id: int, end_word_id: int, eos_id: int) -> np.ndarray:
    """Align fast-tokenizer offsets to rendered word spans."""
    aligned = np.full(enc.length, pad_id, dtype=np.int32)
    text = " ".join(enc.words)
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenized.get("offset_mapping")
    if offsets is None:
        raise ValueError("rendered-text preparation requires a fast tokenizer with offsets")
    word_char_spans = []
    cursor = 0
    for word in enc.words:
        word_char_spans.append((cursor, cursor + len(word)))
        cursor += len(word) + 1
    per_word: list[list[int]] = [[] for _ in enc.words]
    for token, (start, end) in zip(tokenized["input_ids"], offsets):
        for index, (word_start, word_end) in enumerate(word_char_spans):
            if start < word_end and end > word_start:
                per_word[index].append(int(token))
                break
    usable = enc.length - 1
    for index, ((start, end), tokens) in enumerate(zip(enc.word_spans, per_word)):
        marker = 0 if start == 0 else start - 1
        if marker < usable:
            aligned[marker] = end_word_id
        position = max(start, marker + 1)
        next_start = enc.word_spans[index + 1][0] if index + 1 < len(enc.word_spans) else usable
        for token in tokens:
            if position >= min(end, next_start, usable):
                break
            aligned[position] = token
            position += 1
    aligned[-1] = eos_id
    return aligned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_root_metadata(root: Path, image_meta: dict, text_meta: dict, alias: str):
    root.mkdir(parents=True, exist_ok=True)
    expected = {
        root / "meta_img.json": image_meta,
        root / "meta_txt.json": {alias: text_meta},
    }
    for path, payload in expected.items():
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            current = json.loads(path.read_text())
            if path.name == "meta_txt.json" and alias in current:
                current = {alias: current[alias]}
            if current != payload:
                raise ValueError(f"metadata conflict at {path}")
        finally:
            temporary.unlink(missing_ok=True)


def prepare(args) -> Path:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    root = Path(args.output)
    final = root / args.split / f"shard-{args.shard_index:05d}"
    config_path = Path(args.renderer_dir) / "text_renderer_config.json"
    request = {
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "name": args.name,
        "split": args.split,
        "text_column": args.text_column,
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_alias": args.tokenizer_alias,
        "renderer_config_sha256": _sha256(config_path),
        "rgb": args.rgb,
        "max_length": args.max_length,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_samples": args.max_samples,
    }
    if final.exists():
        meta = validate(final)
        if meta.get("request") != request:
            raise ValueError(f"completed shard {final} was prepared with different parameters")
        return final

    renderer_config = json.loads(config_path.read_text())
    font_file = Path(renderer_config["font_file"])
    if not font_file.is_absolute():
        font_file = Path(args.renderer_dir) / font_file
    renderer_config["font_file"] = str(font_file)
    if args.max_length is not None:
        renderer_config["max_seq_length"] = args.max_length
    renderer_config["rgb"] = args.rgb
    renderer = PangoCairoTextRenderer(**renderer_config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id, revision=args.tokenizer_revision, use_fast=True
    )
    base_vocab = tokenizer.vocab_size
    pad_id, end_word_id = base_vocab, base_vocab + 1
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("the text tokenizer must define an EOS token")
    special_vocab = max(base_vocab + 2, eos_id + 1)

    dataset = load_dataset(args.dataset, args.name, split=args.split, revision=args.dataset_revision)
    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards

    temporary = root / args.split / f".shard-{args.shard_index:05d}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    streams = {
        "image.bin": (temporary / "image.bin").open("wb"),
        "image.len": (temporary / "image.len").open("wb"),
        "text.bin": (temporary / "text.bin").open("wb"),
        "samples.jsonl": (temporary / "samples.jsonl").open("w"),
        "skips.jsonl": (temporary / "skips.jsonl").open("w"),
    }
    count = steps = skipped = 0
    try:
        for source_index in range(start, end):
            sample_id = f"{args.dataset}:{args.split}:{source_index}"
            try:
                text = str(dataset[source_index][args.text_column]).strip()
                enc = renderer(text)
                alignment = align_text(enc, tokenizer, pad_id, end_word_id, eos_id)
            except Exception as exc:
                streams["skips.jsonl"].write(
                    json.dumps({"id": sample_id, "reason": type(exc).__name__, "detail": str(exc)}) + "\n"
                )
                skipped += 1
                continue
            enc.pixels.tofile(streams["image.bin"])
            np.asarray([enc.length], dtype=np.int64).tofile(streams["image.len"])
            alignment.tofile(streams["text.bin"])
            streams["samples.jsonl"].write(
                json.dumps({"id": sample_id, "source_index": source_index, "steps": enc.length}) + "\n"
            )
            count += 1
            steps += enc.length
    finally:
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    if not count:
        raise ValueError("refusing to publish an empty rendered-text shard")

    files = list(streams)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "modality": "rendered-text",
        "num_samples": count,
        "num_steps": steps,
        "num_skipped": skipped,
        "n_codebooks": renderer.pixels_per_patch ** 2 * (3 if renderer.rgb else 1),
        "image_vocab_size": 256,
        "image_pad_id": 0,
        "image_eos_id": 0,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "source_range": [start, end],
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_alias": args.tokenizer_alias,
        "base_vocab_size": base_vocab,
        "text_pad_id": pad_id,
        "text_end_word_id": end_word_id,
        "text_eos_id": eos_id,
        "text_special_vocab_size": special_vocab,
        "renderer": renderer_config,
        "renderer_config_sha256": _sha256(config_path),
        "font_sha256": _sha256(font_file),
        "request": request,
        "checksums": {name: _sha256(temporary / name) for name in files},
    }
    (temporary / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    validate(temporary)

    image_meta = {
        "vocab_size": 256,
        "pad_id": 0,
        "eos_id": 0,
        "n_q": meta["n_codebooks"],
        "schema_version": SCHEMA_VERSION,
    }
    text_meta = {
        "tokenizer_id": args.tokenizer_id,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_alias": args.tokenizer_alias,
        "base_vocab_size": base_vocab,
        "pad_id": pad_id,
        "epad_id": end_word_id,
        "eos_id": eos_id,
        "special_vocab_size": special_vocab,
    }
    _write_root_metadata(root, image_meta, text_meta, args.tokenizer_alias)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    return final


def validate(path: str | Path) -> dict:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("modality") != "rendered-text":
        raise ValueError(f"unsupported rendered-text shard: {path}")
    lengths = np.fromfile(path / "image.len", dtype=np.int64)
    if len(lengths) != meta["num_samples"] or np.any(lengths <= 0):
        raise ValueError(f"invalid rendered lengths in {path}")
    steps = int(lengths.sum())
    if steps != meta["num_steps"]:
        raise ValueError(f"rendered step count mismatch in {path}")
    image_bytes = steps * meta["n_codebooks"] * np.dtype(np.int16).itemsize
    if (path / "image.bin").stat().st_size != image_bytes:
        raise ValueError(f"rendered image size mismatch in {path}")
    if (path / "text.bin").stat().st_size != steps * np.dtype(np.int32).itemsize:
        raise ValueError(f"rendered text size mismatch in {path}")
    for name, expected in meta["checksums"].items():
        if _sha256(path / name) != expected:
            raise ValueError(f"checksum mismatch for {path / name}")
    return meta


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
    parser.add_argument("--renderer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--rgb", action="store_true")
    args = parser.parse_args(argv)
    print(prepare(args))


if __name__ == "__main__":
    main()
