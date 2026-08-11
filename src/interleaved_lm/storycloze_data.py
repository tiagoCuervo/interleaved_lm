from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_manifest(source: Path, output_root: Path, task: str) -> Path:
    """Convert a released three-row StoryCloze task manifest to evaluation JSONL."""
    if task not in {"sstorycloze", "tstorycloze"}:
        raise ValueError("task must be sstorycloze or tstorycloze")
    task_root = output_root / task
    if task_root.exists():
        raise FileExistsError(task_root)

    groups: dict[str, dict[str, str]] = defaultdict(dict)
    with source.open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            missing = {"pair_id", "text", "type", "correctness"} - row.keys()
            if missing:
                raise ValueError(f"{source}:{line_number} missing {sorted(missing)}")
            label = "prompt" if row["type"] == "prompt" else row["correctness"]
            if label not in {"prompt", "correct", "incorrect"}:
                raise ValueError(f"{source}:{line_number} has invalid row labels")
            sample_id = str(row["pair_id"])
            if label in groups[sample_id]:
                raise ValueError(f"duplicate {label} row for {sample_id}")
            groups[sample_id][label] = str(row["text"])

    if len(groups) != 1871:
        raise ValueError(f"{task} must contain 1,871 questions, found {len(groups)}")
    if any(set(group) != {"prompt", "correct", "incorrect"} for group in groups.values()):
        raise ValueError("every question needs one prompt, correct ending, and incorrect ending")

    task_root.mkdir(parents=True)
    manifest = task_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for sample_id, group in groups.items():
            stream.write(json.dumps({
                "id": sample_id,
                "prompt": group["prompt"],
                "chosen": group["correct"],
                "rejected": group["incorrect"],
            }, ensure_ascii=False) + "\n")
    metadata = {
        "format_version": 1,
        "task": task,
        "revision": sha256(source),
        "source_name": source.name,
        "source_sha256": sha256(source),
        "num_examples": len(groups),
        "manifest_sha256": sha256(manifest),
    }
    (task_root / "manifest.meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return task_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare one released paper StoryCloze task")
    parser.add_argument("--task", choices=["sstorycloze", "tstorycloze"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    print(prepare_manifest(args.input, args.output, args.task))


if __name__ == "__main__":
    main()
