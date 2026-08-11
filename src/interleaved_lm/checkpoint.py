from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch

from .model import ModelArgs, PerceptionExpressionAdaptedTextLM


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_wrapper_prefixes(state):
    clean = {}
    for name, tensor in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        clean[name] = tensor
    return clean


def load_training_checkpoint(path: str | Path, *, backbone_revision: str | None = None):
    checkpoint = _load(path)
    allowed = {field.name for field in fields(ModelArgs)}
    args = ModelArgs(**{key: value for key, value in checkpoint["model_args"].items() if key in allowed})
    if backbone_revision is not None:
        if args.backbone_revision not in {None, backbone_revision}:
            raise ValueError("requested backbone revision conflicts with checkpoint metadata")
        if args.tokenizer_revision not in {None, backbone_revision}:
            raise ValueError("requested backbone revision conflicts with tokenizer metadata")
        args.backbone_revision = backbone_revision
        args.tokenizer_revision = backbone_revision
    model = PerceptionExpressionAdaptedTextLM(args, is_resume=True)
    model.load_state_dict(strip_wrapper_prefixes(checkpoint["model"]))
    return model.eval()


def convert_checkpoint(
    path: str | Path,
    output: str | Path,
    *,
    backbone_revision: str | None = None,
):
    load_training_checkpoint(
        path, backbone_revision=backbone_revision
    ).save_pretrained(output)
