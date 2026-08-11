"""Download and verify the renderer font from the pinned PIXEL model revision."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


REVISION = "8133b6472dbf8545d1d70078d3cedc20f05e3eae"
URL = f"https://huggingface.co/Team-PIXEL/pixel-base-bigrams/resolve/{REVISION}/GoNotoCurrent.ttf"
SHA256 = "83ab5c39e2b1c34a955136275ce0db068cb20d9643ead033d6b8124a73ab4f64"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "GoNotoCurrent.ttf"
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
        print(destination)
        return
    temporary = destination.with_suffix(".ttf.tmp")
    digest = hashlib.sha256()
    with urlopen(URL) as response, temporary.open("wb") as stream:
        while chunk := response.read(8 << 20):
            stream.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("downloaded font checksum does not match the pinned artifact")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
