"""
Unified any→any evaluator for PerceptionExpressionAdaptedTextLM.

Bins mode (pixenc/hubertenc style)
----------------------------------
If --save_bins is set, we precompute and write dataset-indexed binaries:

data/{dataset}/{pixel|hubert}/
  prompt_{cloze|mc}.bin
  prompt_{cloze|mc}.len
  prompt_{cloze|mc}_{txt_alias}.bin            (aligned text ids per step)

  a_{text|letter}.bin / .len / _{txt_alias}.bin
  b_{text|letter}.bin / .len / _{txt_alias}.bin
  c_{text|letter}.bin / .len / _{txt_alias}.bin
  d_{text|letter}.bin / .len / _{txt_alias}.bin

- Files are dataset-indexed (row i in HF split => i-th entry in .len).
- .bin is a flat 1D stream (int16 for mod tokens, int32 for aligned text).
- We slice via cumulative sums of .len and reshape with n_q when needed.
- Prompts are stored WITHOUT modality EOS (so no internal EOS in assembled sample).
- Continuations are stored WITH EOS (so the sample ends properly).

If binaries exist, we load and use them automatically unless --ignore_bins.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from contextlib import nullcontext
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer

from .model import ModelArgs, PerceptionExpressionAdaptedTextLM
try:
    from .rendered_text import load_renderer_local, image_to_patch_vectors, align_text_to_patches, TOKENIZER_ALIAS
except ImportError:
    TOKENIZER_ALIAS = {
        "HuggingFaceTB/SmolLM-135M": "smollm",
        "HuggingFaceTB/SmolLM-360M": "smollm",
        "HuggingFaceTB/SmolLM-1.7B": "smollm",
    }

    def _missing_renderer(*args, **kwargs):
        raise ImportError("install the rendered-text dependencies for image evaluation")

    load_renderer_local = image_to_patch_vectors = align_text_to_patches = _missing_renderer

try:
    import torchaudio
    from kokoro import KPipeline
except ImportError:
    torchaudio = KPipeline = None

MC_ANSWER_STYLE = "text"  # {"text", "letter"}
DATA_ROOT = Path(os.environ.get("INTERLEAVED_LM_DATA", "data"))



def _valid_opts(dataset: str) -> List[str]:
    if dataset in {"sstorycloze", "tstorycloze"}:
        return ["A", "B"]
    return ["A", "B", "C", "D"]


def build_prompt_and_cont_mmlu(
    sample: Dict,
    prompt_format: str,
) -> Tuple[str, Callable[[str], str]]:
    question = str(sample["question"])
    choices = sample["choices"]
    letters = _valid_opts("mmlu")

    choices_by_L = {
        L: str(choices[idx])
        for idx, L in enumerate(letters)
        if idx < len(choices)
    }

    topic = (sample.get("subject", "general") or "general").replace("_", " ")

    if prompt_format == "mc":
        header = (
            "The following are multiple choice questions (with answers) "
            f"about {topic}.\n\n"
        )
        mc_choices = "\n".join(
            f"{L}. {choices_by_L[L]}" for L in letters if L in choices_by_L
        )

        prompt_txt = header + f"Question: {question}\n{mc_choices}"

        use_text = (MC_ANSWER_STYLE == "text")

        def cont_fn(L: str, choices=choices_by_L):
            if use_text:
                return f"\nAnswer: {choices[L]}"
            else:
                return f"\nAnswer: {L}."

        return prompt_txt, cont_fn

    prompt_txt = (
        "The following are questions about "
        + topic
        + ".\nQuestion: "
        + question
    )

    def cont_fn(L: str, choices=choices_by_L):
        return "\nAnswer: " + choices[L]

    return prompt_txt, cont_fn


def build_prompt_and_cont_hellaswag(
    sample: Dict,
) -> Tuple[str, Callable[[str], str]]:
    stem = str(sample["ctx"])
    endings = sample["endings"]
    letters = _valid_opts("hellaswag")

    choices_by_L = {
        L: str(endings[idx])
        for idx, L in enumerate(letters)
        if idx < len(endings)
    }

    prompt_txt = stem

    def cont_fn(L: str, choices=choices_by_L):
        return choices[L]

    return prompt_txt, cont_fn


def build_prompt_and_cont_storycloze(
    sample: Dict,
) -> Tuple[str, Callable[[str], str]]:
    stem = _normalize_story_text(sample["prompt"])
    endings = [sample["chosen"], sample["rejected"]]
    letters = _valid_opts("sstorycloze")

    choices_by_L = {
        L: _normalize_story_text(endings[idx])
        for idx, L in enumerate(letters)
        if idx < len(endings)
    }

    prompt_txt = stem

    def cont_fn(L: str, choices=choices_by_L):
        return choices[L]

    return prompt_txt, cont_fn


def build_prompt_and_cont(
    dataset: str,
    sample: Dict,
    prompt_format: str,
) -> Tuple[str, Callable[[str], str]]:
    if dataset == "mmlu":
        return build_prompt_and_cont_mmlu(sample, prompt_format)
    if dataset == "hellaswag":
        return build_prompt_and_cont_hellaswag(sample)
    if dataset in {"sstorycloze", "tstorycloze"}:
        return build_prompt_and_cont_storycloze(sample)
    raise ValueError(f"Unknown dataset: {dataset}")


def _normalize_story_text(text: str) -> str:
    return " ".join(str(text).strip().rstrip(".!?").split()).casefold()


def _load_storycloze_manifest(task: str, revision: Optional[str]) -> List[Dict]:
    """Load one explicit paper task; sStoryCloze and tStoryCloze are distinct."""
    root = DATA_ROOT / task
    manifest = root / "manifest.jsonl"
    metadata = root / "manifest.meta.json"
    if not manifest.is_file() or not metadata.is_file():
        raise FileNotFoundError(
            f"{task} requires {manifest} and {metadata}; see DATA.md. "
            "The paper tasks cannot be reconstructed from HF storage splits."
        )

    meta = json.loads(metadata.read_text())
    actual_revision = str(meta.get("revision", ""))
    if not actual_revision:
        raise ValueError(f"{metadata} does not declare a revision")
    if revision is not None and revision != actual_revision:
        raise ValueError(
            f"{task} revision mismatch: requested {revision}, manifest is {actual_revision}"
        )

    rows = []
    seen = set()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "prompt", "chosen", "rejected"} - row.keys()
            if missing:
                raise ValueError(f"{manifest}:{line_number} missing {sorted(missing)}")
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ValueError(f"duplicate StoryCloze id {sample_id!r}")
            if any(not str(row[field]).strip() for field in ("prompt", "chosen", "rejected")):
                raise ValueError(f"{manifest}:{line_number} contains empty text")
            seen.add(sample_id)
            rows.append(dict(row))

    expected = int(meta.get("num_examples", len(rows)))
    if len(rows) != expected:
        raise ValueError(f"{manifest} has {len(rows)} rows; metadata declares {expected}")
    if task in {"sstorycloze", "tstorycloze"} and len(rows) != 1871:
        raise ValueError(f"{task} must contain the paper's 1,871 examples")
    digest = _file_sha256(manifest)
    declared_digest = meta.get("manifest_sha256")
    if declared_digest is not None and digest != declared_digest:
        raise ValueError(f"{manifest} checksum does not match {metadata}")
    return rows



def _text_special_ids(tokenizer):
    base_vocab = tokenizer.vocab_size if tokenizer.vocab_size is not None else len(tokenizer)
    pad_id = base_vocab
    epad_id = base_vocab + 1
    eos_id = tokenizer.eos_token_id
    if eos_id is None or eos_id >= base_vocab or eos_id in (pad_id, epad_id):
        eos_id = base_vocab + 2
    special_vocab_size = max(base_vocab, eos_id + 1, epad_id + 1)
    return base_vocab, pad_id, epad_id, eos_id, special_vocab_size


def _collapse_boundary(p: Optional[np.ndarray], c: Optional[np.ndarray]):
    if p is None or c is None or p.shape[0] == 0 or c.shape[0] == 0:
        return p, c
    if np.array_equal(p[-1], c[0]):
        c = c[1:]
    return p, c


def _pick_shots(i: int, n_total: int, k_shot: int, seed: int) -> np.ndarray:
    if k_shot <= 0:
        return np.empty((0,), dtype=np.int64)
    rng = np.random.RandomState(seed + i)
    pool = np.arange(n_total, dtype=np.int64)
    pool = pool[pool != i]
    if pool.size == 0:
        return np.empty((0,), dtype=np.int64)
    return rng.choice(pool, size=min(k_shot, pool.size), replace=False)



def _part_paths(root: str, part: str, txt_alias: Optional[str]):
    bin_path = os.path.join(root, f"{part}.bin")
    len_path = os.path.join(root, f"{part}.len")
    txt_path = None if txt_alias is None else os.path.join(root, f"{part}_{txt_alias}.bin")
    return bin_path, len_path, txt_path


def _part_complete(root: str, part: str, n_total: int) -> bool:
    _, len_path, _ = _part_paths(root, part, None)
    if not os.path.exists(len_path):
        return False
    lens = np.fromfile(len_path, dtype=np.int64)
    return lens.size == n_total


class BinPartReader:
    def __init__(
        self,
        root: str,
        part: str,
        *,
        n_q: int,
        txt_alias: Optional[str] = None,
        dtype=np.int16,
    ):
        self.root = root
        self.part = part
        self.n_q = int(n_q)
        self.bin_path, self.len_path, self.txt_path = _part_paths(root, part, txt_alias)

        self.lens = np.fromfile(self.len_path, dtype=np.int64)
        self.off = np.zeros((self.lens.size + 1,), dtype=np.int64)
        np.cumsum(self.lens, out=self.off[1:])

        tot_steps = int(self.off[-1])
        tot_mod = tot_steps * self.n_q

        self._mod = np.memmap(self.bin_path, dtype=dtype, mode="r", shape=(tot_mod,))
        self._txt = None
        if self.txt_path is not None and os.path.exists(self.txt_path):
            self._txt = np.memmap(self.txt_path, dtype=np.int32, mode="r", shape=(tot_steps,))

    def get_mod(self, i: int) -> np.ndarray:
        s = int(self.off[i])
        T = int(self.lens[i])
        if T == 0:
            return np.zeros((0, self.n_q), dtype=np.int16)
        a = s * self.n_q
        b = (s + T) * self.n_q
        return self._mod[a:b].reshape(T, self.n_q)

    def get_txt(self, i: int) -> Optional[np.ndarray]:
        if self._txt is None:
            return None
        s = int(self.off[i])
        T = int(self.lens[i])
        return self._txt[s:s + T]


class BinPartWriter:
    def __init__(self, root: str, part: str, *, txt_alias: Optional[str], with_txt: bool):
        self.root = root
        self.part = part
        self.txt_alias = txt_alias
        self.with_txt = with_txt

        os.makedirs(root, exist_ok=True)

        self.bin_path, self.len_path, self.txt_path = _part_paths(root, part, txt_alias)

        self._fb = open(self.bin_path, "wb")
        self._fl = open(self.len_path, "wb")
        self._ft = open(self.txt_path, "wb") if (with_txt and self.txt_path is not None) else None

    def add(self, x16: np.ndarray, w32: Optional[np.ndarray]):
        T = int(x16.shape[0])
        np.array([T], np.int64).tofile(self._fl)
        x16.reshape(-1).astype(np.int16, copy=False).tofile(self._fb)
        if self._ft is not None:
            w32.astype(np.int32, copy=False).reshape(-1).tofile(self._ft)

    def close(self):
        self._fb.close()
        self._fl.close()
        if self._ft is not None:
            self._ft.close()


def _update_meta_txt(save_dir, txt_alias, entry):
    p = os.path.join(save_dir, "meta_txt.json")
    meta = {}
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta[txt_alias] = entry
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)



class ModalitySynthesizer:
    def synthesize_sequence(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_codebooks(self) -> int:
        raise NotImplementedError

    @property
    def pad_token(self) -> int:
        raise NotImplementedError


class TextSynthesizer(ModalitySynthesizer):
    def __init__(self, tokenizer: AutoTokenizer, pad_token_id: int):
        self.tok = tokenizer
        self._pad = pad_token_id
        self._n_codebooks = 1

    @property
    def n_codebooks(self) -> int:
        return self._n_codebooks

    @property
    def pad_token(self) -> int:
        return self._pad

    def synthesize_sequence(self, text: str) -> np.ndarray:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return np.zeros((0, 1), dtype=np.int64)
        return np.array(ids, dtype=np.int64).reshape(-1, 1)


class PixelImageSynthesizer(ModalitySynthesizer):
    def __init__(self, renderer, vocabsize: int, pad_token: int):
        self.renderer = renderer
        self._vocabsize = vocabsize
        self._pad = pad_token
        self._ppb = renderer.pixels_per_patch
        self._rgb = renderer.rgb
        self._n_codebooks = self._ppb * self._ppb * (3 if self._rgb else 1)

    @property
    def n_codebooks(self) -> int:
        return self._n_codebooks

    @property
    def pad_token(self) -> int:
        return self._pad

    def synthesize_sequence(self, text: str) -> np.ndarray:
        enc = self.renderer(text)
        patches = image_to_patch_vectors(enc, ppb=self._ppb, rgb=self._rgb)
        if patches.size == 0:
            return np.zeros((0, self._n_codebooks), dtype=np.int64)
        x16 = np.clip(patches, 0, self._vocabsize - 1).astype(np.int16)
        return x16.astype(np.int64)


MODEL_SR = 16000
MIN_WAV_LEN = 720


class HubertSpeechSynthesizer(torch.nn.Module, ModalitySynthesizer):
    """
    text -> KokoroTTS waveform -> HuBERT -> kmeans codes
    - appends EOS (= n_clusters)
    - collapses adjacent repeats (unique_consecutive)
    """

    _misaki_patched = False

    def __init__(
        self,
        *,
        hubert_model: str,
        hubert_revision: Optional[str],
        hubert_layer: int,
        km_path: str,
        pad_token: int,
        voice: str,
        lang_code: str,
        tts_sr: int = 24000,
        max_chunk: int = 1600000,
        device: str = "cuda",
        txt_tokenizer=None,
        pad_id: Optional[int] = None,
        epad_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ):
        super().__init__()
        self._patch_misaki()

        self.pipe = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.tts_sr = tts_sr

        from interleaved_speech_data.backends import HubertKMeansTokenizer

        self.unit_tokenizer = HubertKMeansTokenizer(
            hubert_model,
            km_path,
            layer=hubert_layer,
            device=device,
            max_seconds=max_chunk // MODEL_SR,
            revision=hubert_revision,
        )
        self.device = torch.device(device)
        self.sample_rate = MODEL_SR
        self.eos_token = self.unit_tokenizer.eos_token
        self._pad = pad_token

        self.txt_tokenizer = txt_tokenizer
        self.pad_id = pad_id
        self.epad_id = epad_id
        self.eos_id = eos_id

        self._cache = {}

    @staticmethod
    def _patch_misaki():
        if HubertSpeechSynthesizer._misaki_patched:
            return
        import misaki.en
        from misaki.en import (
            MToken, replace, subtokenize,
            PUNCT_TAGS, PUNCT_TAG_PHONEMES, PUNCTS, CURRENCIES
        )

        def patched_retokenize(tokens):
            words = []
            currency = None
            for i, token in enumerate(tokens):
                if token._.alias is None and token.phonemes is None:
                    tks = [replace(
                        token, text=t, whitespace='',
                        _=MToken.Underscore(is_head=True, num_flags=token._.num_flags, prespace=False)
                    ) for t in subtokenize(token.text)]
                else:
                    tks = [token]
                tks[-1].whitespace = token.whitespace
                for j, tk in enumerate(tks):
                    if tk._.alias is not None or tk.phonemes is not None:
                        pass
                    elif tk.tag == '$' and tk.text in CURRENCIES:
                        currency = tk.text
                        tk.phonemes = ''
                        tk._.rating = 4
                    elif tk.tag == ':' and tk.text in ('-', '–'):
                        tk.phonemes = '—'
                        tk._.rating = 3
                    elif tk.tag in PUNCT_TAGS and not all(97 <= ord(ch) <= 122 for c in tk.text for ch in c.lower()):
                        tk.phonemes = PUNCT_TAG_PHONEMES.get(tk.tag, ''.join(c for c in tk.text if c in PUNCTS))
                        tk._.rating = 4
                    elif currency is not None:
                        if tk.tag != 'CD':
                            currency = None
                        elif j + 1 == len(tks) and (i + 1 == len(tokens) or tokens[i + 1].tag != 'CD'):
                            tk._.currency = currency
                    elif 0 < j < len(tks) - 1 and tk.text == '2' and (tks[j - 1].text[-1] + tks[j + 1].text[0]).isalpha():
                        tk._.alias = 'to'
                    if tk._.alias is not None or tk.phonemes is not None:
                        words.append(tk)
                    elif words and isinstance(words[-1], list) and not words[-1][-1].whitespace:
                        tk._.is_head = False
                        words[-1].append(tk)
                    else:
                        words.append(tk if tk.whitespace else [tk])
            return [w[0] if isinstance(w, list) and len(w) == 1 else w for w in words]

        misaki.en.G2P.retokenize = staticmethod(patched_retokenize)
        HubertSpeechSynthesizer._misaki_patched = True

    @property
    def n_codebooks(self) -> int:
        return 1

    @property
    def pad_token(self) -> int:
        return self._pad

    @dataclass
    class _Enc:
        num_text_patches: int
        words: List[str]
        word_patch_spans: List[Tuple[int, int]]

    def _tts_words(self, text: str):
        wavs = []
        words = []
        times = []

        t_offset = 0.0
        for result in self.pipe(text, voice=self.voice):
            if hasattr(result, "audio"):
                a = result.audio
                tks = result.tokens
            else:
                a = result[2]
                tks = result[1] if len(result) > 1 else None

            if a is None:
                continue
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            a = np.asarray(a, dtype=np.float32)
            wavs.append(a)

            if tks is not None:
                for t in tks:
                    w = (getattr(t, "text", "") or "").strip()
                    if not w:
                        continue
                    if not hasattr(t, "start_ts") or not hasattr(t, "end_ts"):
                        continue
                    if t.start_ts is None or t.end_ts is None:
                        continue
                    words.append(w)
                    times.append((float(t.start_ts) + t_offset, float(t.end_ts) + t_offset))

            t_offset += a.shape[0] / self.tts_sr

        if not wavs:
            wavs = [np.zeros(int(0.1 * self.tts_sr), np.float32)]

        audio = np.concatenate(wavs).astype(np.float32, copy=False)
        wav = torch.from_numpy(audio).to(self.device)

        if self.tts_sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), self.tts_sr, self.sample_rate).squeeze(0)

        return wav, words, times

    @torch.no_grad()
    def _wav2code_uncollapsed(self, wav: torch.Tensor) -> torch.Tensor:
        waveform = wav.detach().cpu().float().numpy()
        codes = self.unit_tokenizer.encode_uncollapsed(waveform)
        if codes.shape[1] != 1:
            raise ValueError("evaluation speech synthesis requires one K-means codebook")
        return torch.from_numpy(codes[:, 0].astype(np.int64)).to(self.device)

    def _collapse_with_eos(self, codes_unc: torch.Tensor):
        eos = torch.tensor([self.eos_token], device=codes_unc.device, dtype=torch.long)
        x = torch.cat([codes_unc, eos], dim=0)
        keep = torch.ones((x.numel(),), dtype=torch.bool, device=x.device)
        keep[1:] = x[1:] != x[:-1]
        run_id = torch.cumsum(keep, dim=0) - 1
        return x[keep], run_id

    def _aligned_text(self, words, word_times, T, T0, dur, run_id):
        if self.txt_tokenizer is None:
            return None
        if T <= 0:
            return np.zeros((0,), dtype=np.int32)

        usable = T - 1
        if usable <= 0 or not words or not word_times or dur <= 0.0 or T0 <= 0:
            enc = self._Enc(num_text_patches=T, words=[], word_patch_spans=[])
            return align_text_to_patches(enc, self.txt_tokenizer, pad_id=self.pad_id, epad_id=self.epad_id, eos_id=self.eos_id).astype(np.int32)

        w2 = []
        spans = []
        for w, (ts0, ts1) in zip(words, word_times):
            if not (0.0 <= ts0 < ts1 <= dur + 1e-3):
                continue

            s0 = int(np.floor(ts0 / dur * T0))
            s1 = int(np.ceil(ts1 / dur * T0))
            s0 = max(0, min(T0 - 1, s0))
            s1 = max(1, min(T0, s1))
            if s1 <= s0:
                continue

            sc = int(run_id[s0].item())
            ec = int(run_id[s1 - 1].item()) + 1
            if sc >= usable:
                continue
            ec = min(ec, usable)
            if ec <= sc:
                continue

            w2.append(w)
            spans.append((sc, ec))

        enc = self._Enc(num_text_patches=T, words=w2, word_patch_spans=spans)
        return align_text_to_patches(enc, self.txt_tokenizer, pad_id=self.pad_id, epad_id=self.epad_id, eos_id=self.eos_id).astype(np.int32)

    def synthesize_with_txt(self, text: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        out = self._cache.get(text, None)
        if out is not None:
            return out

        wav, words, word_times = self._tts_words(text)
        codes_unc = self._wav2code_uncollapsed(wav)
        codes_coll, run_id = self._collapse_with_eos(codes_unc)

        x16 = codes_coll.detach().cpu().numpy().astype(np.int16).reshape(-1, 1)

        W = None
        if self.txt_tokenizer is not None:
            T = int(x16.shape[0])
            T0 = int(codes_unc.numel())
            dur = float(wav.numel() / self.sample_rate)
            W = self._aligned_text(words, word_times, T, T0, dur, run_id)

        self._cache[text] = (x16, W)
        return x16, W

    def synthesize_sequence(self, text: str) -> np.ndarray:
        x16, _ = self.synthesize_with_txt(text)
        return x16.astype(np.int64)



@dataclass
class Variant:
    input_tokens: torch.Tensor       # (T, C)
    txt_input_mask: torch.Tensor     # (T,)
    img_input_mask: torch.Tensor     # (T,)
    aud_input_mask: torch.Tensor     # (T,)
    txt_preds_mask: torch.Tensor     # (T-1,)
    img_preds_mask: torch.Tensor     # (T-1,)
    aud_preds_mask: torch.Tensor     # (T-1,)


def _collapse_boundary(p: Optional[np.ndarray], c: Optional[np.ndarray]):
    if p is None or c is None or p.shape[0] == 0 or c.shape[0] == 0:
        return p, c
    if np.array_equal(p[-1], c[0]):
        c = c[1:]
    return p, c


def build_any2any_variant(
    model: PerceptionExpressionAdaptedTextLM,
    prompt_text: str,
    cont_text: str,
    in_modality: str,
    out_modality: str,
    txt_synth: TextSynthesizer,
    img_synth: Optional[PixelImageSynthesizer],
    aud_synth: Optional[HubertSpeechSynthesizer],
) -> Variant:
    prompt_img = None
    prompt_aud = None

    if in_modality == "text":
        prompt_txt = txt_synth.synthesize_sequence(prompt_text)

    elif in_modality == "image":
        prompt_img = img_synth.synthesize_sequence(prompt_text)
        if prompt_img.shape[0] > 0:
            prompt_img = prompt_img[:-1]
        Tp = prompt_img.shape[0]
        prompt_txt = np.full((Tp, 1), txt_synth.pad_token, dtype=np.int64)

    elif in_modality == "audio":
        prompt_aud = aud_synth.synthesize_sequence(prompt_text)
        if prompt_aud.shape[0] > 0:
            prompt_aud = prompt_aud[:-1]
        Tp = prompt_aud.shape[0]
        prompt_txt = np.full((Tp, 1), txt_synth.pad_token, dtype=np.int64)

    else:
        raise ValueError(f"in_modality {in_modality} not implemented")

    cont_img = None
    cont_aud = None

    if out_modality == "text":
        cont_txt = txt_synth.synthesize_sequence(cont_text)

    elif out_modality == "image":
        cont_img = img_synth.synthesize_sequence(cont_text)
        Tc = cont_img.shape[0]
        cont_txt = np.full((Tc, 1), txt_synth.pad_token, dtype=np.int64)

    elif out_modality == "audio":
        cont_aud = aud_synth.synthesize_sequence(cont_text)
        Tc = cont_aud.shape[0]
        cont_txt = np.full((Tc, 1), txt_synth.pad_token, dtype=np.int64)

    else:
        raise ValueError(f"out_modality {out_modality} not implemented")

    if in_modality == out_modality == "audio":
        prompt_aud, cont_aud = _collapse_boundary(prompt_aud, cont_aud)
        Tc = 0 if cont_aud is None else cont_aud.shape[0]
        cont_txt = np.full((Tc, 1), txt_synth.pad_token, dtype=np.int64)

    return build_any2any_variant_from_arrays(
        model=model,
        prompt_txt=prompt_txt,
        cont_txt=cont_txt,
        prompt_img=prompt_img,
        cont_img=cont_img,
        prompt_aud=prompt_aud,
        cont_aud=cont_aud,
        in_modality=in_modality,
        out_modality=out_modality,
    )


def build_any2any_variant_from_arrays(
    *,
    model: PerceptionExpressionAdaptedTextLM,
    prompt_txt: np.ndarray,
    cont_txt: np.ndarray,
    prompt_img: Optional[np.ndarray],
    cont_img: Optional[np.ndarray],
    prompt_aud: Optional[np.ndarray],
    cont_aud: Optional[np.ndarray],
    in_modality: str,
    out_modality: str,
) -> Variant:
    Tp = prompt_txt.shape[0]
    Tc = cont_txt.shape[0]

    use_swt = (in_modality != out_modality) and (model.swt_token is not None)
    T = Tp + Tc + (1 if use_swt else 0)

    C = model.n_codebooks
    tokens = np.zeros((T, C), dtype=np.int64)

    if model.models_txt and model.txt_pad_token is not None:
        tokens[:, 0] = model.txt_pad_token
    if model.models_img and model.img_pad_token is not None:
        tokens[:, model.img_slice] = model.img_pad_token
    if model.models_aud and model.aud_pad_token is not None:
        tokens[:, model.aud_slice] = model.aud_pad_token

    tokens[:Tp, 0] = prompt_txt[:, 0]

    swt_row = Tp if use_swt else None
    cont_start = Tp + (1 if use_swt else 0)

    if use_swt:
        tokens[swt_row, 0] = model.swt_token

    tokens[cont_start:cont_start + Tc, 0] = cont_txt[:, 0]

    if model.n_img_codebooks > 0:
        base = model.img_slice.start
        k = model.n_img_codebooks
        if prompt_img is not None and prompt_img.shape[0] > 0:
            tokens[:Tp, base:base + k] = prompt_img[:, :k]
        if cont_img is not None and cont_img.shape[0] > 0:
            tokens[cont_start:cont_start + cont_img.shape[0], base:base + k] = cont_img[:, :k]

    if model.n_aud_codebooks > 0:
        base = model.aud_slice.start
        if prompt_aud is not None and prompt_aud.shape[0] > 0:
            tokens[:Tp, base] = prompt_aud[:, 0]
        if cont_aud is not None and cont_aud.shape[0] > 0:
            tokens[cont_start:cont_start + cont_aud.shape[0], base] = cont_aud[:, 0]

    T = tokens.shape[0]

    txt_input_mask = (
        (tokens[:, 0] != model.txt_pad_token)
        if model.models_txt else np.zeros(T, bool)
    )

    if model.models_img:
        img_channel = tokens[:, model.img_slice]
        img_input_mask = (img_channel != model.img_pad_token).any(axis=-1)
    else:
        img_input_mask = np.zeros(T, bool)

    if model.models_aud:
        aud_channel = tokens[:, model.aud_slice]
        aud_input_mask = (aud_channel != model.aud_pad_token).any(axis=-1)
    else:
        aud_input_mask = np.zeros(T, bool)

    txt_preds_mask = np.zeros(T - 1, bool)
    img_preds_mask = np.zeros(T - 1, bool)
    aud_preds_mask = np.zeros(T - 1, bool)

    first_target_in_idx = cont_start - 1

    if out_modality == "text" and model.models_txt:
        txt_preds_mask[first_target_in_idx:T - 1] = True
    if out_modality == "image" and model.models_img:
        img_preds_mask[first_target_in_idx:T - 1] = True
    if out_modality == "audio" and model.models_aud:
        aud_preds_mask[first_target_in_idx:T - 1] = True

    return Variant(
        input_tokens=torch.tensor(tokens, dtype=torch.long),
        txt_input_mask=torch.tensor(txt_input_mask, dtype=torch.bool),
        img_input_mask=torch.tensor(img_input_mask, dtype=torch.bool),
        aud_input_mask=torch.tensor(aud_input_mask, dtype=torch.bool),
        txt_preds_mask=torch.tensor(txt_preds_mask, dtype=torch.bool),
        img_preds_mask=torch.tensor(img_preds_mask, dtype=torch.bool),
        aud_preds_mask=torch.tensor(aud_preds_mask, dtype=torch.bool),
    )


def collate_variants(var_list: List[Variant], pad_tok: int, n_codebooks: int):
    input_tokens = [v.input_tokens for v in var_list]
    txt_in = [v.txt_input_mask for v in var_list]
    img_in = [v.img_input_mask for v in var_list]
    aud_in = [v.aud_input_mask for v in var_list]
    txt_pred = [v.txt_preds_mask for v in var_list]
    img_pred = [v.img_preds_mask for v in var_list]
    aud_pred = [v.aud_preds_mask for v in var_list]

    padded_tokens = pad_sequence(input_tokens, batch_first=True, padding_value=pad_tok)
    if padded_tokens.size(-1) != n_codebooks:
        raise RuntimeError(
            f"padded_tokens last dim {padded_tokens.size(-1)} != n_codebooks {n_codebooks}"
        )

    def pad_bool(seqs):
        return pad_sequence(seqs, batch_first=True, padding_value=False)

    return {
        "input_tokens_bsc": padded_tokens,
        "txt_input_masks_bs": pad_bool(txt_in),
        "img_input_masks_bs": pad_bool(img_in),
        "aud_input_masks_bs": pad_bool(aud_in),
        "txt_preds_masks_bs_in": pad_bool(txt_pred),
        "img_preds_masks_bs_in": pad_bool(img_pred),
        "aud_preds_masks_bs_in": pad_bool(aud_pred),
    }



def _cont_suffix(prompt_format: str, mc_answer_style: str) -> str:
    return mc_answer_style if prompt_format == "mc" else "text"


def _write_meta_parts(root: str, ds_key: str, parts: List[str], extra: Dict):
    p = os.path.join(root, "meta_eval_mc.json")
    meta = {}
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
    meta.update({
        "dataset": ds_key,
        "parts": sorted(set(parts)),
    })
    meta.update(extra)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _ensure_pixel_bins(
    *,
    ds,
    ds_key: str,
    n_total: int,
    valid_opts: List[str],
    prompt_format: str,
    mc_answer_style: str,
    k_shot: int,
    shot_seed: int,
    renderer_dir: str,
    renderer_max_length: Optional[int],
    renderer_rgb: bool,
    txt_tokenizer,
    pad_id: int,
    epad_id: int,
    eos_id: int,
    base_vocab: int,
    special_vocab_size: int,
    img_vocab: int,
    img_pad: int,
    need_prompt: bool,
    need_cont: bool,
    txt_alias: Optional[str],
    ignore_bins: bool,
):
    root = os.path.join(DATA_ROOT, ds_key, "pixel")
    parts = []
    prompt_part = f"prompt_{prompt_format}"
    suff = _cont_suffix(prompt_format, mc_answer_style)
    cont_parts = {L: f"{L.lower()}_{suff}" for L in valid_opts}

    want_txt = (txt_alias is not None)

    if ignore_bins:
        for part in [prompt_part] + list(cont_parts.values()):
            for p in _part_paths(root, part, txt_alias):
                if p is not None and os.path.exists(p):
                    os.remove(p)

    need_build = False
    if need_prompt and not _part_complete(root, prompt_part, n_total):
        need_build = True
    if need_cont:
        for part in cont_parts.values():
            if not _part_complete(root, part, n_total):
                need_build = True
                break

    if not need_build:
        return

    if txt_alias is None:
        raise ValueError(f"missing TOKENIZER_ALIAS for {txt_tokenizer.name_or_path}")

    renderer_kwargs = {}
    if renderer_max_length is not None:
        renderer_kwargs["max_seq_length"] = renderer_max_length
    renderer_kwargs["rgb"] = renderer_rgb

    renderer, renderer_cfg = load_renderer_local(renderer_dir, **renderer_kwargs)
    ppb = renderer.pixels_per_patch
    rgb = renderer.rgb
    n_q = ppb * ppb * (3 if rgb else 1)

    parts_to_write = []
    if need_prompt:
        parts_to_write.append(prompt_part)
    if need_cont:
        parts_to_write.extend(cont_parts.values())

    writers = {part: BinPartWriter(root, part, txt_alias=txt_alias, with_txt=want_txt) for part in parts_to_write}

    for i in tqdm(range(n_total), desc=f"build-pixel-{ds_key}", leave=False):
        ex = ds[i]

        shot_idx = _pick_shots(i, n_total, k_shot, shot_seed)
        shot_txt = ""
        for j in shot_idx:
            s_ex = ds[int(j)]
            s_prompt, s_cont_fn = build_prompt_and_cont(ds_key.split("_")[0], s_ex, prompt_format=prompt_format)

            if ds_key.startswith("mmlu_"):
                letters = _valid_opts("mmlu")
                s_gold = str(s_ex["answer"]).strip()
                if s_gold not in letters:
                    continue
            elif ds_key == "hellaswag":
                s_gold = _valid_opts("hellaswag")[int(s_ex["label"])]
            elif ds_key in {"sstorycloze", "tstorycloze"}:
                s_gold = _valid_opts(ds_key)[0]
            else:
                s_gold = None

            s_cont_txt = s_cont_fn(s_gold)
            shot_txt += s_prompt + s_cont_txt + "\n\n"

        prompt_txt_main, cont_fn_main = build_prompt_and_cont(ds_key.split("_")[0], ex, prompt_format=prompt_format)
        full_prompt_txt = shot_txt + prompt_txt_main

        if need_prompt:
            enc = renderer(full_prompt_txt)
            patches = image_to_patch_vectors(enc, ppb=ppb, rgb=rgb)
            x16 = np.clip(patches, 0, img_vocab - 1).astype(np.int16)
            W = align_text_to_patches(enc, txt_tokenizer, pad_id, epad_id, eos_id).astype(np.int32)

            if x16.shape[0] > 0:
                x16 = x16[:-1]
                W = W[:-1]

            writers[prompt_part].add(x16, W)
            parts.append(prompt_part)

        if need_cont:
            for L, part in cont_parts.items():
                cont_txt = cont_fn_main(L)
                enc = renderer(cont_txt)
                patches = image_to_patch_vectors(enc, ppb=ppb, rgb=rgb)
                x16 = np.clip(patches, 0, img_vocab - 1).astype(np.int16)
                W = align_text_to_patches(enc, txt_tokenizer, pad_id, epad_id, eos_id).astype(np.int32)
                writers[part].add(x16, W)
                parts.append(part)

    for w in writers.values():
        w.close()

    _update_meta_txt(root, txt_alias, {
        "dataset": ds_key,
        "text_column": "eval_mc",
        "tokenizer_id": txt_tokenizer.name_or_path,
        "tokenizer_alias": txt_alias,
        "base_vocab_size": base_vocab,
        "pad_id": pad_id,
        "epad_id": epad_id,
        "eos_id": eos_id,
        "special_vocab_size": special_vocab_size,
    })
    with open(os.path.join(root, "meta_img.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset": ds_key,
            "text_column": "eval_mc",
            "vocab_size": img_vocab,
            "rgb": rgb,
            "n_q": n_q,
            "renderer_config": renderer_cfg,
            "pad_token": int(img_pad),
        }, f, indent=2)

    _write_meta_parts(root, ds_key, parts, {
        "prompt_format": prompt_format,
        "mc_answer_style": mc_answer_style,
        "k_shot": k_shot,
        "shot_seed": shot_seed,
    })


def _ensure_hubert_bins(
    *,
    ds,
    ds_key: str,
    n_total: int,
    valid_opts: List[str],
    prompt_format: str,
    mc_answer_style: str,
    k_shot: int,
    shot_seed: int,
    txt_tokenizer,
    pad_id: int,
    epad_id: int,
    eos_id: int,
    base_vocab: int,
    special_vocab_size: int,
    hubert_model: str,
    hubert_revision: Optional[str],
    hubert_km_path: str,
    hubert_layer: int,
    hubert_max_chunk: int,
    kokoro_voice: str,
    kokoro_lang_code: str,
    kokoro_sr: int,
    aud_pad: int,
    device: str,
    need_prompt: bool,
    need_cont: bool,
    txt_alias: Optional[str],
    ignore_bins: bool,
):
    root = os.path.join(DATA_ROOT, ds_key, "hubert")
    parts = []
    prompt_part = f"prompt_{prompt_format}"
    suff = _cont_suffix(prompt_format, mc_answer_style)
    cont_parts = {L: f"{L.lower()}_{suff}" for L in valid_opts}

    want_txt = (txt_alias is not None)

    if ignore_bins:
        for part in [prompt_part] + list(cont_parts.values()):
            for p in _part_paths(root, part, txt_alias):
                if p is not None and os.path.exists(p):
                    os.remove(p)

    need_build = False
    if need_prompt and not _part_complete(root, prompt_part, n_total):
        need_build = True
    if need_cont:
        for part in cont_parts.values():
            if not _part_complete(root, part, n_total):
                need_build = True
                break

    if not need_build:
        return

    if txt_alias is None:
        raise ValueError(f"missing TOKENIZER_ALIAS for {txt_tokenizer.name_or_path}")

    aud = HubertSpeechSynthesizer(
        hubert_model=hubert_model,
        hubert_revision=hubert_revision,
        hubert_layer=hubert_layer,
        km_path=hubert_km_path,
        pad_token=aud_pad,
        voice=kokoro_voice,
        lang_code=kokoro_lang_code,
        tts_sr=kokoro_sr,
        max_chunk=hubert_max_chunk,
        device=device,
        txt_tokenizer=txt_tokenizer,
        pad_id=pad_id,
        epad_id=epad_id,
        eos_id=eos_id,
    ).to(device)

    parts_to_write = []
    if need_prompt:
        parts_to_write.append(prompt_part)
    if need_cont:
        parts_to_write.extend(cont_parts.values())

    writers = {part: BinPartWriter(root, part, txt_alias=txt_alias, with_txt=want_txt) for part in parts_to_write}

    for i in tqdm(range(n_total), desc=f"build-hubert-{ds_key}", leave=False):
        ex = ds[i]

        shot_idx = _pick_shots(i, n_total, k_shot, shot_seed)
        shot_txt = ""
        for j in shot_idx:
            s_ex = ds[int(j)]
            s_prompt, s_cont_fn = build_prompt_and_cont(ds_key.split("_")[0], s_ex, prompt_format=prompt_format)

            if ds_key.startswith("mmlu_"):
                letters = _valid_opts("mmlu")
                s_gold = str(s_ex["answer"]).strip()
                if s_gold not in letters:
                    continue
            elif ds_key == "hellaswag":
                s_gold = _valid_opts("hellaswag")[int(s_ex["label"])]
            elif ds_key in {"sstorycloze", "tstorycloze"}:
                s_gold = _valid_opts(ds_key)[0]
            else:
                s_gold = None

            s_cont_txt = s_cont_fn(s_gold)
            shot_txt += s_prompt + s_cont_txt + "\n\n"

        prompt_txt_main, cont_fn_main = build_prompt_and_cont(ds_key.split("_")[0], ex, prompt_format=prompt_format)
        full_prompt_txt = shot_txt + prompt_txt_main

        if need_prompt:
            x16, W = aud.synthesize_with_txt(full_prompt_txt)
            if x16.shape[0] > 0:
                x16 = x16[:-1]
                W = W[:-1]
            writers[prompt_part].add(x16, W)
            parts.append(prompt_part)

        if need_cont:
            for L, part in cont_parts.items():
                cont_txt = cont_fn_main(L)
                x16, W = aud.synthesize_with_txt(cont_txt)
                writers[part].add(x16, W)
                parts.append(part)

    for w in writers.values():
        w.close()

    _update_meta_txt(root, txt_alias, {
        "dataset": ds_key,
        "text_column": "eval_mc",
        "tokenizer_id": txt_tokenizer.name_or_path,
        "tokenizer_alias": txt_alias,
        "base_vocab_size": base_vocab,
        "pad_id": pad_id,
        "epad_id": epad_id,
        "eos_id": eos_id,
        "special_vocab_size": special_vocab_size,
    })
    with open(os.path.join(root, "meta_aud.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset": ds_key,
            "text_column": "eval_mc",
            "hubert_model": hubert_model,
            "hubert_revision": hubert_revision,
            "hubert_layer": hubert_layer,
            "km_path": hubert_km_path,
            "tts_voice": kokoro_voice,
            "tts_lang_code": kokoro_lang_code,
            "tts_sr": kokoro_sr,
            "sr_hubert": 16000,
            "n_q": 1,
            "vocab_size": int(aud.eos_token) + 1,
            "eos_token": int(aud.eos_token),
            "pad_token": int(aud.pad_token),
            "collapsed": True,
        }, f, indent=2)

    _write_meta_parts(root, ds_key, parts, {
        "prompt_format": prompt_format,
        "mc_answer_style": mc_answer_style,
        "k_shot": k_shot,
        "shot_seed": shot_seed,
    })



def _eval_core(
    *,
    model: PerceptionExpressionAdaptedTextLM,
    backbone_id: str,
    text_tokenizer,
    eval_dataset: str,
    dataset_revision: Optional[str],
    mmlu_subject: str,
    prompt_format: str,
    mc_answer_style: str,
    in_modality: str,
    out_modality: str,
    batch_size: int,
    eval_fraction: float,
    renderer_dir: Optional[str],
    renderer_max_length: Optional[int],
    renderer_rgb: bool,
    k_shot: int,
    shot_seed: int,
    ctx,
    show_progress: bool,
    hubert_model: Optional[str],
    hubert_revision: Optional[str],
    hubert_km_path: Optional[str],
    hubert_layer: int,
    hubert_max_chunk: int,
    kokoro_voice: str,
    kokoro_lang_code: str,
    kokoro_sr: int,
    save_bins: bool,
    ignore_bins: bool,
    use_bins: bool,
) -> Tuple[float, List[Dict]]:
    global MC_ANSWER_STYLE
    MC_ANSWER_STYLE = mc_answer_style

    if eval_dataset == "mmlu":
        if mmlu_subject is None:
            raise ValueError("mmlu_subject is required when eval_dataset='mmlu'")
        raw = load_dataset(
            "cais/mmlu", mmlu_subject, split="test", revision=dataset_revision
        )
        letters = ["A", "B", "C", "D"]
        ds = [
            {
                "question": ex["question"],
                "choices": ex["choices"],
                "answer": letters[int(ex["answer"])],
                "subject": ex.get("subject", mmlu_subject),
            }
            for ex in raw
        ]
    elif eval_dataset == "hellaswag":
        ds = load_dataset("hellaswag", split="validation", revision=dataset_revision)
    elif eval_dataset in {"sstorycloze", "tstorycloze"}:
        ds = _load_storycloze_manifest(eval_dataset, dataset_revision)
    else:
        raise ValueError(f"Unknown eval_dataset: {eval_dataset}")

    rng = np.random.RandomState(shot_seed)
    n_total = len(ds)
    eval_indices = np.arange(n_total)
    if eval_fraction < 1.0:
        n_eval = max(1, int(round(eval_fraction * n_total)))
        eval_indices = rng.choice(eval_indices, size=n_eval, replace=False)

    print(
        f"[INFO] [{eval_dataset} {in_modality}->{out_modality}] "
        f"Evaluating on {len(eval_indices)}/{n_total} examples "
        f"({len(eval_indices) / max(n_total, 1):.2%})"
    )

    txt_tokenizer = text_tokenizer
    if txt_tokenizer is None:
        txt_tokenizer = AutoTokenizer.from_pretrained(
            backbone_id,
            revision=model.config.tokenizer_revision or model.config.backbone_revision,
        )
    txt_pad = model.txt_pad_token
    if txt_pad is None:
        txt_pad = txt_tokenizer.pad_token_id
    if txt_pad is None:
        txt_pad = txt_tokenizer.eos_token_id
    txt_synth = TextSynthesizer(txt_tokenizer, pad_token_id=txt_pad)

    base_vocab, pad_id, epad_id, eos_id, special_vocab_size = _text_special_ids(txt_tokenizer)
    txt_alias = TOKENIZER_ALIAS.get(backbone_id, None)

    ds_key = eval_dataset if eval_dataset != "mmlu" else f"mmlu_{mmlu_subject}"
    valid_opts = _valid_opts(eval_dataset)
    suff = _cont_suffix(prompt_format, mc_answer_style)
    prompt_part = f"prompt_{prompt_format}"
    cont_parts = {L: f"{L.lower()}_{suff}" for L in valid_opts}

    dev = next(model.parameters()).device

    needs_image = ((in_modality == "image") or (out_modality == "image")) and getattr(model, "models_img", False)
    needs_audio = ((in_modality == "audio") or (out_modality == "audio")) and getattr(model, "models_aud", False)

    if save_bins:
        if txt_alias is None:
            raise ValueError(f"missing TOKENIZER_ALIAS for {backbone_id}")
        if needs_image:
            if renderer_dir is None:
                raise RuntimeError("renderer_dir is required for image bins")
            img_vocab = getattr(model, "img_vocabsize", None) or 256
            img_pad = model.img_pad_token if model.img_pad_token is not None else 0
            _ensure_pixel_bins(
                ds=ds,
                ds_key=ds_key,
                n_total=n_total,
                valid_opts=valid_opts,
                prompt_format=prompt_format,
                mc_answer_style=mc_answer_style,
                k_shot=k_shot,
                shot_seed=shot_seed,
                renderer_dir=renderer_dir,
                renderer_max_length=renderer_max_length,
                renderer_rgb=renderer_rgb,
                txt_tokenizer=txt_tokenizer,
                pad_id=pad_id,
                epad_id=epad_id,
                eos_id=eos_id,
                base_vocab=base_vocab,
                special_vocab_size=special_vocab_size,
                img_vocab=img_vocab,
                img_pad=img_pad,
                need_prompt=(in_modality == "image"),
                need_cont=(out_modality == "image"),
                txt_alias=txt_alias,
                ignore_bins=ignore_bins,
            )
        if needs_audio:
            if hubert_model is None or hubert_km_path is None:
                raise RuntimeError("hubert_model and hubert_km_path required for audio bins")
            _ensure_hubert_bins(
                ds=ds,
                ds_key=ds_key,
                n_total=n_total,
                valid_opts=valid_opts,
                prompt_format=prompt_format,
                mc_answer_style=mc_answer_style,
                k_shot=k_shot,
                shot_seed=shot_seed,
                txt_tokenizer=txt_tokenizer,
                pad_id=pad_id,
                epad_id=epad_id,
                eos_id=eos_id,
                base_vocab=base_vocab,
                special_vocab_size=special_vocab_size,
                hubert_model=hubert_model,
                hubert_revision=hubert_revision,
                hubert_km_path=hubert_km_path,
                hubert_layer=hubert_layer,
                hubert_max_chunk=hubert_max_chunk,
                kokoro_voice=kokoro_voice,
                kokoro_lang_code=kokoro_lang_code,
                kokoro_sr=kokoro_sr,
                aud_pad=model.aud_pad_token,
                device=str(dev),
                need_prompt=(in_modality == "audio"),
                need_cont=(out_modality == "audio"),
                txt_alias=txt_alias,
                ignore_bins=ignore_bins,
            )

    img_prompt_reader = None
    img_cont_readers = None
    aud_prompt_reader = None
    aud_cont_readers = None
    txt_prompt_reader = None
    txt_cont_readers = None

    if use_bins and not ignore_bins:
        root = os.path.join(DATA_ROOT, ds_key, "text")
        if in_modality == "text" and k_shot == 0 and _part_complete(root, prompt_part, n_total):
            txt_prompt_reader = BinPartReader(root, prompt_part, n_q=1, dtype=np.int32)
        if out_modality == "text":
            if all(_part_complete(root, part, n_total) for part in cont_parts.values()):
                txt_cont_readers = {
                    letter: BinPartReader(root, cont_parts[letter], n_q=1, dtype=np.int32)
                    for letter in valid_opts
                }

        if needs_image:
            root = os.path.join(DATA_ROOT, ds_key, "pixel")
            meta_p = os.path.join(root, "meta_img.json")
            if os.path.exists(meta_p):
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta_img = json.load(f)
                n_q = int(meta_img["n_q"])
            else:
                n_q = None

            if n_q is not None:
                if (in_modality == "image") and _part_complete(root, prompt_part, n_total):
                    img_prompt_reader = BinPartReader(root, prompt_part, n_q=n_q, txt_alias=txt_alias)
                if out_modality == "image":
                    ok = True
                    for part in cont_parts.values():
                        if not _part_complete(root, part, n_total):
                            ok = False
                            break
                    if ok:
                        img_cont_readers = {L: BinPartReader(root, cont_parts[L], n_q=n_q, txt_alias=txt_alias) for L in valid_opts}

        if needs_audio:
            root = os.path.join(DATA_ROOT, ds_key, "hubert")
            if (in_modality == "audio") and _part_complete(root, prompt_part, n_total):
                aud_prompt_reader = BinPartReader(root, prompt_part, n_q=1, txt_alias=txt_alias)
            if out_modality == "audio":
                ok = True
                for part in cont_parts.values():
                    if not _part_complete(root, part, n_total):
                        ok = False
                        break
                if ok:
                    aud_cont_readers = {L: BinPartReader(root, cont_parts[L], n_q=1, txt_alias=txt_alias) for L in valid_opts}

    img_synth = None
    needs_runtime_image = (
        (in_modality == "image" and img_prompt_reader is None)
        or (out_modality == "image" and img_cont_readers is None)
    )
    if needs_runtime_image:
        if renderer_dir is None:
            raise RuntimeError("renderer_dir is required when using image modality")
        renderer_kwargs = {}
        if renderer_max_length is not None:
            renderer_kwargs["max_seq_length"] = renderer_max_length
        renderer_kwargs["rgb"] = renderer_rgb
        renderer, _ = load_renderer_local(renderer_dir, **renderer_kwargs)
        img_vocab = getattr(model, "img_vocabsize", None) or 256
        img_pad = model.img_pad_token if model.img_pad_token is not None else 0
        img_synth = PixelImageSynthesizer(renderer, vocabsize=img_vocab, pad_token=img_pad)

    aud_synth = None
    needs_runtime_audio = (
        (in_modality == "audio" and aud_prompt_reader is None)
        or (out_modality == "audio" and aud_cont_readers is None)
    )
    if needs_runtime_audio:
        if hubert_model is None or hubert_km_path is None:
            raise RuntimeError("hubert_model and hubert_km_path required for audio modality")
        aud_synth = HubertSpeechSynthesizer(
            hubert_model=hubert_model,
            hubert_revision=hubert_revision,
            hubert_layer=hubert_layer,
            km_path=hubert_km_path,
            pad_token=model.aud_pad_token,
            voice=kokoro_voice,
            lang_code=kokoro_lang_code,
            tts_sr=kokoro_sr,
            max_chunk=hubert_max_chunk,
            device=str(dev),
        ).to(dev)

    correct = 0
    kept = 0
    row_records: List[Dict] = []
    B = batch_size

    was_training = model.training
    model.eval()

    try:
        iterator = range(0, len(eval_indices), B)
        if show_progress:
            iterator = tqdm(iterator, desc=f"eval-{eval_dataset}-{in_modality}->{out_modality}")

        for start in iterator:
            idx_batch = eval_indices[start:start + B]
            variants_per_option = {L: [] for L in valid_opts}
            gold_letters = []
            row_idx_list = []

            for idx in idx_batch:
                i = int(idx)
                ex = ds[i]

                if eval_dataset == "mmlu":
                    gold_letter = str(ex["answer"]).strip()
                elif eval_dataset == "hellaswag":
                    gold_letter = valid_opts[int(ex["label"])]
                elif eval_dataset in {"sstorycloze", "tstorycloze"}:
                    gold_letter = valid_opts[0]
                else:
                    raise ValueError

                if gold_letter not in valid_opts:
                    continue

                shot_txt = ""
                if k_shot > 0:
                    pick = _pick_shots(i, n_total, k_shot, shot_seed)
                    for j in pick:
                        s_ex = ds[int(j)]
                        s_prompt, s_cont_fn = build_prompt_and_cont(eval_dataset, s_ex, prompt_format=prompt_format)
                        if eval_dataset == "mmlu":
                            s_gold = str(s_ex["answer"]).strip()
                        elif eval_dataset == "hellaswag":
                            s_gold = valid_opts[int(s_ex["label"])]
                        elif eval_dataset in {"sstorycloze", "tstorycloze"}:
                            s_gold = valid_opts[0]
                        else:
                            raise ValueError
                        if s_gold not in valid_opts:
                            continue
                        shot_txt += s_prompt + s_cont_fn(s_gold) + "\n\n"

                prompt_txt_main, cont_fn_main = build_prompt_and_cont(eval_dataset, ex, prompt_format=prompt_format)
                full_prompt_txt = shot_txt + prompt_txt_main

                gold_letters.append(gold_letter)
                row_idx_list.append(i)

                prompt_img = None
                prompt_aud = None
                if in_modality == "image":
                    if img_prompt_reader is not None:
                        prompt_img = np.array(img_prompt_reader.get_mod(i), copy=False)
                    else:
                        prompt_img = img_synth.synthesize_sequence(full_prompt_txt)
                        if prompt_img.shape[0] > 0:
                            prompt_img = prompt_img[:-1]
                if in_modality == "audio":
                    if aud_prompt_reader is not None:
                        prompt_aud = np.array(aud_prompt_reader.get_mod(i), copy=False)
                    else:
                        prompt_aud = aud_synth.synthesize_sequence(full_prompt_txt)
                        if prompt_aud.shape[0] > 0:
                            prompt_aud = prompt_aud[:-1]


                if in_modality == "text":
                    if txt_prompt_reader is not None:
                        prompt_txt_ids0 = np.array(txt_prompt_reader.get_mod(i), copy=False)
                    else:
                        prompt_txt_ids0 = txt_synth.synthesize_sequence(full_prompt_txt)
                elif in_modality == "image":
                    Tp0 = 0 if prompt_img is None else prompt_img.shape[0]
                    prompt_txt_ids0 = np.full((Tp0, 1), txt_synth.pad_token, dtype=np.int64)
                elif in_modality == "audio":
                    Tp0 = 0 if prompt_aud is None else prompt_aud.shape[0]
                    prompt_txt_ids0 = np.full((Tp0, 1), txt_synth.pad_token, dtype=np.int64)
                else:
                    raise ValueError

                for L in valid_opts:
                    cont_txt = cont_fn_main(L)

                    cont_img = None
                    cont_aud = None

                    if out_modality == "image":
                        if img_cont_readers is not None:
                            cont_img = np.array(img_cont_readers[L].get_mod(i), copy=False)
                        else:
                            cont_img = img_synth.synthesize_sequence(cont_txt)

                    if out_modality == "audio":
                        if aud_cont_readers is not None:
                            cont_aud = np.array(aud_cont_readers[L].get_mod(i), copy=False)
                        else:
                            cont_aud = aud_synth.synthesize_sequence(cont_txt)

                    pa = prompt_aud
                    ca = cont_aud
                    if in_modality == out_modality == "audio":
                        pa, ca = _collapse_boundary(prompt_aud, cont_aud)

                    if out_modality == "text":
                        if txt_cont_readers is not None:
                            cont_txt_ids = np.array(txt_cont_readers[L].get_mod(i), copy=False)
                        else:
                            cont_txt_ids = txt_synth.synthesize_sequence(cont_txt)
                    elif out_modality == "image":
                        Tc = 0 if cont_img is None else cont_img.shape[0]
                        cont_txt_ids = np.full((Tc, 1), txt_synth.pad_token, dtype=np.int64)
                    elif out_modality == "audio":
                        Tc = 0 if ca is None else ca.shape[0]
                        cont_txt_ids = np.full((Tc, 1), txt_synth.pad_token, dtype=np.int64)
                    else:
                        raise ValueError

                    var = build_any2any_variant_from_arrays(
                        model=model,
                        prompt_txt=prompt_txt_ids0,
                        cont_txt=cont_txt_ids,
                        prompt_img=prompt_img if in_modality == "image" else None,
                        cont_img=cont_img if out_modality == "image" else None,
                        prompt_aud=pa if in_modality == "audio" else None,
                        cont_aud=ca if out_modality == "audio" else None,
                        in_modality=in_modality,
                        out_modality=out_modality,
                    )
                    variants_per_option[L].append(var)

            bs = len(gold_letters)
            kept += bs
            if bs == 0:
                continue

            best_lp = torch.full((bs,), -1e30, device=dev)
            best_opt_idx = torch.full_like(best_lp, -1, dtype=torch.long)
            row_lp = [[None] * len(valid_opts) for _ in range(bs)]

            for j, L in enumerate(valid_opts):
                vars_L = variants_per_option[L]
                if len(vars_L) == 0:
                    continue

                batch = collate_variants(vars_L, pad_tok=0, n_codebooks=model.n_codebooks)
                batch = {k: v.to(dev) for k, v in batch.items()}

                with torch.no_grad(), ctx:
                    lp = model.log_likelihood(
                        **batch,
                        txt_weight=1.0,
                        img_codebook_weights=None,
                        aud_codebook_weights=None,
                    )

                for k_idx, score in enumerate(lp.tolist()):
                    row_lp[k_idx][j] = score
                mask = lp > best_lp
                best_lp[mask] = lp[mask]
                best_opt_idx[mask] = j

            gold_idx = torch.tensor([valid_opts.index(x) for x in gold_letters], device=dev)
            correct += (best_opt_idx == gold_idx).sum().item()

            for k_idx in range(bs):
                row_records.append({
                    "index": row_idx_list[k_idx],
                    "gold": gold_letters[k_idx],
                    "pred": valid_opts[best_opt_idx[k_idx].item()] if best_opt_idx[k_idx] >= 0 else None,
                    "logprobs": row_lp[k_idx],
                })
    finally:
        model.train(was_training)

    acc = correct / kept if kept else 0.0
    print(f"[RESULT] [{eval_dataset} {in_modality}->{out_modality}] Accuracy: {acc:.4%} ({correct}/{kept})")
    return acc, row_records



def run_eval(
    *,
    model: PerceptionExpressionAdaptedTextLM,
    backbone_id: str,
    eval_datasets: List[str],
    eval_modalities: List[Tuple[str, str]],
    dataset_revisions: Dict[str, str],
    batch_size: int = 8,
    eval_fraction: float = 1.0,
    renderer_dir: Optional[str] = None,
    k_shot: int = 0,
    shot_seed: int = 1337,
    prompt_format: str = "mc",
    mc_answer_style: str = "text",
    mmlu_subject: str = "all",
    renderer_max_length: Optional[int] = None,
    renderer_rgb: bool = False,
    hubert_model: Optional[str] = None,
    hubert_revision: Optional[str] = None,
    hubert_km_path: Optional[str] = None,
    hubert_layer: int = 11,
    hubert_max_chunk: int = 1600000,
    kokoro_voice: str = "af_heart",
    kokoro_lang_code: str = "a",
    kokoro_sr: int = 24000,
    save_bins: bool = False,
    ignore_bins: bool = False,
    use_bins: bool = True,
    precision: str = "bfloat16",
) -> Dict[str, float]:
    dev = next(model.parameters()).device
    if dev.type == "cuda":
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
        ctx = nullcontext() if dtype == torch.float32 else torch.amp.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = nullcontext()

    metrics: Dict[str, float] = {}

    for dset in eval_datasets:
        if dset not in dataset_revisions:
            raise ValueError(f"missing immutable dataset revision for {dset}")
        for in_mod, out_mod in eval_modalities:
            name = f"{dset}_{in_mod}_in_{out_mod}_out_{prompt_format}"
            if prompt_format == "mc":
                name += f"_{mc_answer_style}ans"

            acc, _ = _eval_core(
                model=model,
                backbone_id=backbone_id,
                text_tokenizer=model.global_workspace.txt_tokenizer,
                eval_dataset=dset,
                dataset_revision=dataset_revisions[dset],
                mmlu_subject=mmlu_subject,
                prompt_format=prompt_format,
                mc_answer_style=mc_answer_style,
                in_modality=in_mod,
                out_modality=out_mod,
                batch_size=batch_size,
                eval_fraction=eval_fraction,
                renderer_dir=renderer_dir,
                renderer_max_length=renderer_max_length,
                renderer_rgb=renderer_rgb,
                k_shot=k_shot,
                shot_seed=shot_seed,
                ctx=ctx,
                show_progress=False,
                hubert_model=hubert_model,
                hubert_revision=hubert_revision,
                hubert_km_path=hubert_km_path,
                hubert_layer=hubert_layer,
                hubert_max_chunk=hubert_max_chunk,
                kokoro_voice=kokoro_voice,
                kokoro_lang_code=kokoro_lang_code,
                kokoro_sr=kokoro_sr,
                save_bins=save_bins,
                ignore_bins=ignore_bins,
                use_bins=use_bins,
            )
            metrics[name] = acc

    return metrics



def _load_model(path: str, device: str):
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        return (
            PerceptionExpressionAdaptedTextLM.from_pretrained(checkpoint_path, device=device),
            None,
            checkpoint_path,
        )
    if not checkpoint_path.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("install huggingface-hub to load a remote model") from exc
        resolved = Path(snapshot_download(path))
        return (
            PerceptionExpressionAdaptedTextLM.from_pretrained(resolved, device=device),
            None,
            resolved,
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    allowed = {field.name for field in dataclasses.fields(ModelArgs)}
    model_args = ModelArgs(**{key: value for key, value in checkpoint["model_args"].items() if key in allowed})
    model = PerceptionExpressionAdaptedTextLM(model_args, is_resume=True)
    state = checkpoint["model"]
    prefixes = ("_orig_mod.", "module.")
    clean = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        clean[key] = value
    model.load_state_dict(clean)
    return model.to(device).eval(), checkpoint, checkpoint_path


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hashes(path: Path) -> Dict[str, str]:
    if path.is_file():
        return {path.name: _file_sha256(path)}
    if not path.is_dir():
        return {}
    files = [path / "config.json", *sorted(path.glob("model*.safetensors"))]
    index = path / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
    return {file.name: _file_sha256(file) for file in files if file.exists()}


def _prepared_data_hashes(args) -> Dict[str, str]:
    dataset = args.eval_dataset
    if dataset == "mmlu":
        dataset = f"mmlu_{args.mmlu_subject}"
    dataset_root = Path(args.data_root) / dataset
    suffix = _cont_suffix(args.prompt_format, args.mc_answer_style)
    files = [dataset_root / "manifest.jsonl", dataset_root / "manifest.meta.json"]
    roots = {"text": "text", "image": "pixel", "audio": "hubert"}
    metadata = {
        "text": ("meta_txt.json", "meta_eval_mc.json"),
        "image": ("meta_img.json", "meta_eval_mc.json"),
        "audio": ("meta_aud.json", "meta_eval_mc.json"),
    }
    for modality in sorted({args.in_modality, args.out_modality}):
        root = dataset_root / roots[modality]
        parts = []
        if args.in_modality == modality:
            parts.append(f"prompt_{args.prompt_format}")
        if args.out_modality == modality:
            parts.extend(
                f"{letter.lower()}_{suffix}"
                for letter in _valid_opts(args.eval_dataset)
            )
        for part in parts:
            files.extend((root / f"{part}.bin", root / f"{part}.len"))
        files.extend(root / name for name in metadata[modality])
    return {
        str(path.relative_to(dataset_root)): _file_sha256(path)
        for path in files
        if path.exists()
    }


def main(argv: Optional[List[str]] = None):
    global DATA_ROOT
    ap = argparse.ArgumentParser(description="Evaluate text, rendered-text, and speech directions")
    ap.add_argument(
        "--eval_dataset",
        choices=["mmlu", "hellaswag", "sstorycloze", "tstorycloze"],
        default="mmlu",
    )
    ap.add_argument(
        "--dataset_revision",
        required=True,
        help="Immutable dataset or prepared task revision used for evaluation.",
    )
    ap.add_argument("--mmlu_subject", type=str, default="all")
    ap.add_argument("--prompt_format", choices=["cloze", "mc"], default="cloze")
    ap.add_argument("--mc_answer_style", choices=["text", "letter"], default="text")
    ap.add_argument("--in_modality", choices=["text", "image", "audio"], default="text")
    ap.add_argument("--out_modality", choices=["text", "image", "audio"], default="text")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data_root", default="data")

    ap.add_argument("--k_shot", type=int, default=0)
    ap.add_argument("--shot_seed", type=int, default=1337)
    ap.add_argument("--eval_fraction", type=float, default=1.0)

    ap.add_argument("--renderer_dir", type=str, default="./renderer")
    ap.add_argument("--renderer_max_length", type=int, default=None)
    ap.add_argument("--renderer_rgb", action="store_true")

    ap.add_argument("--hubert_model", type=str)
    ap.add_argument("--hubert_revision", type=str)
    ap.add_argument("--hubert_km_path", type=str)
    ap.add_argument("--hubert_layer", type=int, default=11)
    ap.add_argument("--hubert_max_chunk", type=int, default=1600000)

    ap.add_argument("--kokoro_voice", type=str, default="af_heart")
    ap.add_argument("--kokoro_lang_code", type=str, default="a")
    ap.add_argument("--kokoro_sr", type=int, default=24000)

    ap.add_argument("--save_bins", action="store_true")
    ap.add_argument("--ignore_bins", action="store_true")
    ap.add_argument("--no_use_bins", action="store_true", help="Disable loading precomputed bins.")

    args = ap.parse_args(argv)

    if not (0.0 < args.eval_fraction <= 1.0):
        raise ValueError("--eval_fraction must be in (0, 1].")

    DATA_ROOT = Path(args.data_root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctx_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model, checkpoint, resolved_checkpoint = _load_model(args.checkpoint, dev)
    ctx = nullcontext() if dev == "cpu" or ctx_dtype == torch.float32 else torch.amp.autocast(device_type=dev, dtype=ctx_dtype)

    acc, row_records = _eval_core(
        model=model,
        backbone_id=model.config.backbone,
        text_tokenizer=model.global_workspace.txt_tokenizer,
        eval_dataset=args.eval_dataset,
        dataset_revision=args.dataset_revision,
        mmlu_subject=args.mmlu_subject,
        prompt_format=args.prompt_format,
        mc_answer_style=args.mc_answer_style,
        in_modality=args.in_modality,
        out_modality=args.out_modality,
        batch_size=args.batch_size,
        eval_fraction=args.eval_fraction,
        renderer_dir=args.renderer_dir,
        renderer_max_length=args.renderer_max_length,
        renderer_rgb=args.renderer_rgb,
        k_shot=args.k_shot,
        shot_seed=args.shot_seed,
        ctx=ctx,
        show_progress=True,
        hubert_model=args.hubert_model,
        hubert_revision=args.hubert_revision,
        hubert_km_path=args.hubert_km_path,
        hubert_layer=args.hubert_layer,
        hubert_max_chunk=args.hubert_max_chunk,
        kokoro_voice=args.kokoro_voice,
        kokoro_lang_code=args.kokoro_lang_code,
        kokoro_sr=args.kokoro_sr,
        save_bins=args.save_bins,
        ignore_bins=args.ignore_bins,
        use_bins=(not args.no_use_bins),
    )

    variant_key = f"{args.in_modality}_in_{args.out_modality}_out_{args.prompt_format}"
    if args.prompt_format == "mc":
        variant_key += f"_{args.mc_answer_style}ans"
    if args.k_shot > 0:
        variant_key += f"_{args.k_shot}shot"
    if args.eval_fraction < 1.0:
        variant_key += f"_frac{args.eval_fraction:.2f}"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "format_version": 1,
        "task": args.eval_dataset,
        "dataset_revision": args.dataset_revision,
        "direction": {"input": args.in_modality, "output": args.out_modality},
        "prompt_format": args.prompt_format,
        "answer_style": args.mc_answer_style,
        "accuracy": acc,
        "num_examples": len(row_records),
        "batch_size": args.batch_size,
        "precision": args.dtype,
        "seed": args.shot_seed,
        "checkpoint": args.checkpoint,
        "checkpoint_files_sha256": _checkpoint_hashes(resolved_checkpoint),
        "prepared_data_files_sha256": _prepared_data_hashes(args),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    row_dump_path = output.with_suffix(".rows.jsonl")
    with row_dump_path.open("w") as f:
        for rec in row_records:
            f.write(json.dumps(rec) + "\n")
    result["rows_file"] = {
        "name": row_dump_path.name,
        "sha256": _file_sha256(row_dump_path),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
