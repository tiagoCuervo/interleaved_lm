from __future__ import annotations

import numpy as np
import torch


def text_prompt(model, text: str, device: str | torch.device | None = None):
    if not model.models_txt:
        raise ValueError("this checkpoint has no text input stream")
    if not model.models_aud:
        raise ValueError("this checkpoint has no audio stream")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    text_ids = model.global_workspace.txt_tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    length = text_ids.shape[1]
    tokens = torch.empty((1, length, model.n_codebooks), dtype=torch.long, device=device)
    tokens[..., 0] = text_ids
    if model.models_img:
        tokens[..., model.img_slice] = model.img_pad_token
    tokens[..., model.aud_slice] = model.aud_pad_token
    txt_mask = torch.ones((1, length), dtype=torch.bool, device=device)
    img_mask = torch.zeros_like(txt_mask)
    aud_mask = torch.zeros_like(txt_mask)
    return tokens, txt_mask, img_mask, aud_mask


@torch.inference_mode()
def generate_speech(
    model,
    text: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = None,
) -> torch.Tensor:
    prompt, txt_mask, img_mask, aud_mask = text_prompt(model, text)
    output = model.generate(
        prompt,
        txt_mask,
        img_mask,
        aud_mask,
        max_new_tokens=max_new_tokens,
        gen_aud=True,
        temperature=temperature,
        top_k=top_k,
    )
    codes = output[:, prompt.shape[1] + 1 :, model.aud_slice]
    return codes


def load_codehifigan(
    device: str | torch.device = "cpu",
    vocab_size: int = 500,
    *,
    cache_dir: str | None = None,
):
    if vocab_size != 500:
        raise ValueError("the released mHuBERT CodeHiFiGAN vocoder requires 500 units")
    from .vocoder import load_vocoder

    return load_vocoder(device, cache_dir=cache_dir)


@torch.inference_mode()
def decode_hubert(codes, vocoder, *, vocab_size: int = 500) -> np.ndarray:
    tokens = torch.as_tensor(codes, dtype=torch.long, device=next(vocoder.parameters()).device)
    if tokens.ndim == 3:
        if tokens.shape[-1] != 1:
            raise ValueError("CodeHiFiGAN only decodes one-codebook HuBERT checkpoints")
        tokens = tokens[..., 0]
    tokens = tokens.reshape(-1)
    tokens = tokens[(tokens >= 0) & (tokens < vocab_size)].reshape(1, -1)
    if not tokens.numel():
        raise ValueError("no decodable HuBERT units")
    return vocoder(tokens, dur_prediction=True).squeeze().float().cpu().numpy()
