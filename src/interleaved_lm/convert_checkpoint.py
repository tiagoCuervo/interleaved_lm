import argparse

from .checkpoint import convert_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert an LF2AR training checkpoint to native safetensors")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument(
        "--backbone-revision",
        help="immutable HF revision required when absent from the source checkpoint",
    )
    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint,
        args.output,
        backbone_revision=args.backbone_revision,
    )


if __name__ == "__main__":
    main()
