from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from .dataset import DatasetArgs, MultimodalPretrainDataset, Task
from .model import ModelArgs, PerceptionExpressionAdaptedTextLM


@dataclass
class TrainSettings:
    output_dir: str = "out"
    resume: str | None = None
    seed: int = 1337
    batch_size: int = 2
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    eval_interval: int = 100
    eval_steps: int = 10
    log_interval: int = 10
    save_interval: int = 100
    backbone_lr: float = 2e-4
    adapter_lr: float = 2e-3
    min_backbone_lr: float = 2e-5
    min_adapter_lr: float = 2e-4
    warmup_steps: int = 100
    warmdown_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bfloat16"
    distributed: str = "ddp"
    fsdp_cpu_offload: bool = False
    compile: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def load_config(path: str | Path) -> tuple[str, DatasetArgs, DatasetArgs, ModelArgs, TrainSettings, dict]:
    payload = json.loads(Path(path).read_text())
    data_root = payload.get("data_root", "data")
    train_data = DatasetArgs(data_root=data_root, **payload["train_data"])
    val_data = DatasetArgs(data_root=data_root, **payload["validation_data"])
    settings = TrainSettings(**payload.get("training", {}))
    allowed = {field.name for field in fields(ModelArgs)}
    unknown = set(payload["model"]) - allowed
    if unknown:
        raise ValueError(f"unknown model fields: {sorted(unknown)}")
    model_args = ModelArgs(**payload["model"])
    return data_root, train_data, val_data, model_args, settings, payload


def _distributed(mode: str):
    rank = int(os.environ.get("RANK", "-1"))
    if rank < 0:
        return False, False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode not in {"ddp", "fsdp"}:
        raise ValueError("distributed must be ddp or fsdp")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return True, mode == "fsdp", rank, local_rank, world_size, device


def _unwrap(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def _without_compile(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _model_args_from_data(base: ModelArgs, dataset: MultimodalPretrainDataset) -> ModelArgs:
    values = asdict(base)
    values.update(
        txt_vocabsize=dataset.txt_vocabsize,
        img_vocabsize=dataset.img_vocabsize or None,
        aud_vocabsize=dataset.aud_vocabsize or None,
        txt_pad_token=dataset.txt_pad_token,
        img_pad_token=dataset.img_pad_token,
        aud_pad_token=dataset.aud_pad_token,
        swt_token=dataset.txt_swt_token,
        n_img_codebooks=dataset.img_ncodebooks,
        n_aud_codebooks=dataset.aud_ncodebooks,
        block_size=dataset.config.block_size,
    )
    return ModelArgs(**values)


def _check_validation_data(
    train: MultimodalPretrainDataset,
    validation: MultimodalPretrainDataset,
) -> None:
    fields_to_match = (
        "txt_vocabsize",
        "img_vocabsize",
        "aud_vocabsize",
        "txt_pad_token",
        "img_pad_token",
        "aud_pad_token",
        "txt_swt_token",
        "img_ncodebooks",
        "aud_ncodebooks",
    )
    mismatches = {
        name: (getattr(train, name), getattr(validation, name))
        for name in fields_to_match
        if getattr(train, name) != getattr(validation, name)
    }
    if mismatches:
        raise ValueError(f"training and validation modality metadata differ: {mismatches}")
    if validation.config.block_size > train.config.block_size:
        raise ValueError("validation block_size cannot exceed training block_size")


def _make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, precision: str):
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
    return nullcontext() if device.type == "cpu" or dtype == torch.float32 else torch.autocast(device.type, dtype=dtype)


def _schedule(step: int, peak: float, floor: float, settings: TrainSettings) -> float:
    if step < settings.warmup_steps:
        return peak * (step + 1) / max(1, settings.warmup_steps)
    decay_start = settings.max_steps - settings.warmdown_steps
    if step < decay_start:
        return peak
    ratio = min(1.0, (step - decay_start + 1) / max(1, settings.warmdown_steps))
    if ratio == 1.0:
        return floor
    return peak + ratio * (floor - peak)


def _tag_optimizer_groups(optimizer, raw_model):
    tags = {}
    for name, parameter in raw_model.named_parameters():
        tags[id(parameter)] = "img" if "img_" in name else "aud" if "aud_" in name else "backbone"
    for group in optimizer.param_groups:
        group_tags = [tags[id(parameter)] for parameter in group["params"]]
        group["modality"] = max(set(group_tags), key=group_tags.count)


def _set_learning_rates(optimizer, step: int, settings: TrainSettings):
    backbone = _schedule(step, settings.backbone_lr, settings.min_backbone_lr, settings)
    adapter = _schedule(step, settings.adapter_lr, settings.min_adapter_lr, settings)
    for group in optimizer.param_groups:
        group["lr"] = backbone if group["modality"] == "backbone" else adapter
    return backbone, adapter


def _rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _collect_rng(rank: int, world_size: int):
    local = _rng_state()
    if not dist.is_initialized():
        return [local]
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


_EXACT_RESUME_FIELDS = (
    "seed",
    "batch_size",
    "num_workers",
    "gradient_accumulation_steps",
    "max_steps",
    "eval_interval",
    "eval_steps",
    "backbone_lr",
    "adapter_lr",
    "min_backbone_lr",
    "min_adapter_lr",
    "warmup_steps",
    "warmdown_steps",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
    "precision",
    "distributed",
    "fsdp_cpu_offload",
    "compile",
)


def _validate_checkpoint_model(checkpoint: dict[str, Any], model_args: ModelArgs) -> None:
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported or missing checkpoint format_version")
    if checkpoint.get("model_args") != asdict(model_args):
        raise ValueError("checkpoint model configuration does not match")


def _validate_resume_settings(checkpoint: dict[str, Any], settings: TrainSettings) -> None:
    previous = checkpoint.get("training")
    if not isinstance(previous, dict):
        raise ValueError("resume checkpoint is missing its training configuration")
    mismatches = {
        name: (previous.get(name), getattr(settings, name))
        for name in _EXACT_RESUME_FIELDS
        if previous.get(name) != getattr(settings, name)
    }
    if mismatches:
        raise ValueError(f"resume training configuration differs: {mismatches}")


def _checkpoint_state(model, optimizer, is_fsdp: bool):
    if not is_fsdp:
        return _unwrap(model).state_dict(), optimizer.state_dict()
    model = _without_compile(model)
    policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, policy):
        model_state = model.state_dict()
        optimizer_state = FullyShardedDataParallel.optim_state_dict(model, optimizer)
    return model_state, optimizer_state


def _save(
    path,
    model,
    optimizer,
    scaler,
    is_fsdp,
    step,
    model_args,
    payload,
    settings,
    rng_by_rank,
    world_size,
    dataset_fingerprint,
):
    model_state, optimizer_state = _checkpoint_state(model, optimizer, is_fsdp)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    checkpoint = {
        "format_version": 1,
        "model": model_state,
        "optimizer": optimizer_state,
        "scaler": scaler.state_dict(),
        "step": step,
        "model_args": asdict(model_args),
        "experiment": payload,
        "training": asdict(settings),
        "batch_provenance": _batch_provenance(settings, model_args.block_size, world_size),
        "rng_by_rank": rng_by_rank,
        "world_size": world_size,
        "dataset_fingerprint": dataset_fingerprint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def _batch_provenance(settings: TrainSettings, context_tokens: int, world_size: int) -> dict[str, int]:
    return {
        "micro_batch_sequences_per_rank": settings.batch_size,
        "micro_steps_per_rank": settings.gradient_accumulation_steps // world_size,
        "world_size": world_size,
        "global_sequences_per_update": settings.batch_size * settings.gradient_accumulation_steps,
        "context_tokens": context_tokens,
        "tokens_per_update": (
            settings.batch_size * settings.gradient_accumulation_steps * context_tokens
        ),
    }


def _unpack(batch):
    return batch


@torch.inference_mode()
def _evaluate(model, dataset, settings, device, world_size):
    model.eval()
    dataset.set_iteration(seed=settings.seed + 100_000, start_index=0, rank=dist.get_rank() if dist.is_initialized() else 0)
    batches = Task.iter_batches(settings.batch_size, device, settings.num_workers, dataset)
    total = torch.zeros(2, device=device)
    for _ in range(settings.eval_steps):
        batch = next(batches)
        with _autocast(device, settings.precision):
            model(*_unpack(batch))
            loss = _unwrap(model).last_loss.detach().float()
        total += torch.stack((loss, torch.ones((), device=device)))
    if dist.is_initialized():
        dist.all_reduce(total)
    model.train()
    return (total[0] / total[1]).item()


def train(config_path: str | Path) -> dict[str, float]:
    _, train_args, val_args, base_model_args, settings, payload = load_config(config_path)
    is_distributed, is_fsdp, rank, local_rank, world_size, device = _distributed(settings.distributed)
    master = rank == 0
    if settings.gradient_accumulation_steps % world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by world size")
    accumulation = settings.gradient_accumulation_steps // world_size
    seed = settings.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset = MultimodalPretrainDataset(train_args)
    val_dataset = MultimodalPretrainDataset(val_args)
    _check_validation_data(train_dataset, val_dataset)
    dataset_fingerprint = {
        "train": train_dataset.fingerprint(),
        "validation": val_dataset.fingerprint(),
    }
    model_args = _model_args_from_data(base_model_args, train_dataset)
    resume_checkpoint = _torch_load(settings.resume) if settings.resume else None
    model = PerceptionExpressionAdaptedTextLM(
        model_args, is_resume=resume_checkpoint is not None
    )
    start_step = 0
    if resume_checkpoint:
        _validate_checkpoint_model(resume_checkpoint, model_args)
        model.load_state_dict(resume_checkpoint["model"])
        _validate_resume_settings(resume_checkpoint, settings)
        start_step = int(resume_checkpoint["step"])
        if resume_checkpoint["world_size"] != world_size:
            raise ValueError("exact resume requires the original world size")
        if resume_checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("resume dataset fingerprint does not match")
        _restore_rng(resume_checkpoint["rng_by_rank"][rank])
    model.to(device)

    if is_fsdp:
        layer_type = type(model.global_workspace.context_model.layers[0])
        wrap_policy = lambda_auto_wrap(layer_type)
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[settings.precision]
        mixed = None if dtype == torch.float32 else MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed,
            cpu_offload=CPUOffload(offload_params=settings.fsdp_cpu_offload),
            device_id=device if device.type == "cuda" else None,
            use_orig_params=True,
        )
    elif is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    if settings.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile=True requires PyTorch 2 or newer")
        model = torch.compile(model)

    raw_model = _unwrap(model)
    optimizer = raw_model.configure_optimizers(
        settings.weight_decay,
        settings.backbone_lr,
        (settings.beta1, settings.beta2),
        device.type,
        img_learning_rate=settings.adapter_lr,
        aud_learning_rate=settings.adapter_lr,
    )
    _tag_optimizer_groups(optimizer, raw_model)
    scaler = _make_scaler(device.type == "cuda" and settings.precision == "float16")
    if resume_checkpoint:
        optim_state = resume_checkpoint.get("optimizer")
        if optim_state:
            if is_fsdp:
                optim_state = FullyShardedDataParallel.optim_state_dict_to_load(
                    _without_compile(model), optimizer, optim_state
                )
            optimizer.load_state_dict(optim_state)
        scaler.load_state_dict(resume_checkpoint.get("scaler", {}))

    samples_per_step = settings.batch_size * accumulation
    train_dataset.set_iteration(
        seed=settings.seed,
        start_index=start_step * samples_per_step,
        rank=rank,
    )
    train_batches = Task.iter_batches(settings.batch_size, device, settings.num_workers, train_dataset)
    output = Path(settings.output_dir)
    if master:
        output.mkdir(parents=True, exist_ok=True)
        runtime_config = dict(payload)
        runtime_config["batch_provenance"] = _batch_provenance(
            settings, model_args.block_size, world_size
        )
        (output / "config.json").write_text(json.dumps(runtime_config, indent=2) + "\n")
    wandb_run = None
    if master and settings.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=settings.wandb_project,
            name=settings.wandb_run_name,
            config={
                **payload,
                "batch_provenance": _batch_provenance(
                    settings, model_args.block_size, world_size
                ),
            },
        )

    last_loss = float("nan")
    started = time.perf_counter()
    try:
        model.train()
        for step in range(start_step, settings.max_steps):
            backbone_lr, adapter_lr = _set_learning_rates(optimizer, step, settings)
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            for micro_step in range(accumulation):
                distributed_model = _without_compile(model)
                if isinstance(distributed_model, DistributedDataParallel):
                    distributed_model.require_backward_grad_sync = micro_step == accumulation - 1
                batch = next(train_batches)
                with _autocast(device, settings.precision):
                    model(*_unpack(batch))
                    loss = _unwrap(model).last_loss / accumulation
                scaler.scale(loss).backward()
                last_loss = float(loss.detach()) * accumulation
            if settings.grad_clip:
                scaler.unscale_(optimizer)
                if is_fsdp:
                    _without_compile(model).clip_grad_norm_(settings.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_started
            completed = step + 1
            if master and (completed == 1 or completed % settings.log_interval == 0):
                mfu = raw_model.estimate_mfu(settings.batch_size * accumulation, elapsed)
                metrics = {
                    "step": completed,
                    "loss/train": last_loss,
                    "lr/backbone": backbone_lr,
                    "lr/adapter": adapter_lr,
                    "mfu/raw": mfu,
                    "step_time": elapsed,
                }
                print(json.dumps(metrics))
                if wandb_run:
                    wandb_run.log(metrics, step=completed)
            if completed % settings.eval_interval == 0 or completed == settings.max_steps:
                val_loss = _evaluate(model, val_dataset, settings, device, world_size)
                if master:
                    metrics = {"step": completed, "loss/validation": val_loss}
                    print(json.dumps(metrics))
                    if wandb_run:
                        wandb_run.log(metrics, step=completed)
            if completed % settings.save_interval == 0 or completed == settings.max_steps:
                rng_by_rank = _collect_rng(rank, world_size)
                _save(
                    output / f"checkpoint-{completed:08d}.pt",
                    model,
                    optimizer,
                    scaler,
                    is_fsdp,
                    completed,
                    model_args,
                    payload,
                    settings,
                    rng_by_rank,
                    world_size,
                    dataset_fingerprint,
                )
                if dist.is_initialized():
                    dist.barrier()
    finally:
        if wandb_run:
            wandb_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
    return {"loss": last_loss, "elapsed": time.perf_counter() - started}


def lambda_auto_wrap(layer_type):
    from functools import partial

    return partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_type})


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train LF2AR with DDP or FSDP")
    parser.add_argument("config")
    args = parser.parse_args(argv)
    train(args.config)


if __name__ == "__main__":
    main()
