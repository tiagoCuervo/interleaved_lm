import csv

import pytest

from interleaved_lm.storycloze_data import prepare_manifest


def _write_source(path, count=1871):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["pair_id", "text", "type", "correctness"],
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow({"pair_id": index, "text": "Story", "type": "prompt", "correctness": "-"})
            writer.writerow({"pair_id": index, "text": "Good", "type": "continuation", "correctness": "correct"})
            writer.writerow({"pair_id": index, "text": "Bad", "type": "continuation", "correctness": "incorrect"})


def test_prepare_storycloze_manifest(tmp_path):
    source = tmp_path / "source.csv"
    _write_source(source)
    root = prepare_manifest(source, tmp_path / "data", "sstorycloze")
    assert len((root / "manifest.jsonl").read_text().splitlines()) == 1871
    assert '"num_examples": 1871' in (root / "manifest.meta.json").read_text()


def test_prepare_storycloze_rejects_incomplete_release(tmp_path):
    source = tmp_path / "source.csv"
    _write_source(source, count=1)
    with pytest.raises(ValueError, match="1,871"):
        prepare_manifest(source, tmp_path / "data", "tstorycloze")
