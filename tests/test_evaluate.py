import hashlib
import json

import numpy as np
import pytest

import interleaved_lm.evaluate as evaluate
from interleaved_lm.evaluate import (
    BinPartReader,
    _load_storycloze_manifest,
    _valid_opts,
    build_prompt_and_cont,
)


def test_storycloze_uses_separate_paper_tasks_and_verbatim_boundary():
    sample = {
        "prompt": "A four-sentence story.",
        "chosen": "The fitting ending.",
        "rejected": "An unrelated ending!",
    }
    for task in ("sstorycloze", "tstorycloze"):
        prompt, continuation = build_prompt_and_cont(task, sample, prompt_format="cloze")
        assert _valid_opts(task) == ["A", "B"]
        assert prompt == "a four-sentence story"
        assert continuation("A") == "the fitting ending"
        assert continuation("B") == "an unrelated ending"


def test_storycloze_requires_explicit_revisioned_manifest(tmp_path, monkeypatch):
    root = tmp_path / "sstorycloze"
    root.mkdir()
    rows = [
        {"id": str(i), "prompt": "Story.", "chosen": "Good.", "rejected": f"Bad {i}."}
        for i in range(1871)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    manifest = root / "manifest.jsonl"
    manifest.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (root / "manifest.meta.json").write_text(json.dumps({
        "revision": "paper-v1",
        "num_examples": 1871,
        "manifest_sha256": digest,
    }))
    monkeypatch.setattr(evaluate, "DATA_ROOT", tmp_path)

    assert len(_load_storycloze_manifest("sstorycloze", "paper-v1")) == 1871
    with pytest.raises(ValueError, match="revision mismatch"):
        _load_storycloze_manifest("sstorycloze", "wrong")
    manifest.write_text(payload + "\n")
    with pytest.raises(ValueError, match="checksum"):
        _load_storycloze_manifest("sstorycloze", "paper-v1")


def test_prepared_text_bins_preserve_ids_above_int16(tmp_path):
    np.asarray([3], dtype=np.int64).tofile(tmp_path / "prompt_cloze.len")
    np.asarray([7, 40_000, 49_151], dtype=np.int32).tofile(tmp_path / "prompt_cloze.bin")
    reader = BinPartReader(str(tmp_path), "prompt_cloze", n_q=1, dtype=np.int32)
    assert reader.get_mod(0)[:, 0].tolist() == [7, 40_000, 49_151]
