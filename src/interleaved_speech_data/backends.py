from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
import hashlib
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def load_audio(path: str | Path, *, offset: float = 0.0, duration: float | None = None, target_rate: int = 16000):
    """Load a mono segment with SoundFile and resample it once when needed."""
    import soundfile as sf

    info = sf.info(path)
    start = round(offset * info.samplerate)
    frames = -1 if duration is None else round(duration * info.samplerate)
    waveform, sample_rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != target_rate:
        try:
            from torchaudio.functional import resample
        except ImportError as exc:
            raise ImportError("resampling requires torchaudio; source audio is not 16 kHz") from exc
        waveform = resample(
            torch.from_numpy(np.asarray(waveform)).view(1, -1),
            int(sample_rate),
            target_rate,
        ).view(-1).numpy()
    return np.asarray(waveform, dtype=np.float32), target_rate


def collapse_codes_with_mapping(
    codes: np.ndarray, eos_token: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse consecutive code vectors and map every input frame to its output run."""
    codes = np.asarray(codes)
    if codes.ndim == 1:
        codes = codes[:, None]
    eos = np.full((1, codes.shape[1]), eos_token, dtype=codes.dtype)
    codes = np.concatenate((codes, eos), axis=0)
    keep = np.ones(len(codes), dtype=bool)
    keep[1:] = np.any(codes[1:] != codes[:-1], axis=1)
    frame_to_code = np.cumsum(keep, dtype=np.int64) - 1
    return codes[keep].astype(np.int16), frame_to_code


def collapse_codes(codes: np.ndarray, eos_token: int) -> np.ndarray:
    return collapse_codes_with_mapping(codes, eos_token)[0]


class HubertKMeansTokenizer:
    """Hugging Face mHuBERT features followed by one or more K-means codebooks."""

    def __init__(
        self,
        model: str,
        kmeans: str | Sequence[str],
        revision: str,
        layer: int = 11,
        device: str = "cuda",
        max_seconds: int = 30,
        normalize: bool | None = None,
    ):
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("install the interleaved-speech-data hubert dependencies") from exc
        try:
            from transformers import AutoFeatureExtractor, HubertModel
        except ImportError as exc:
            raise ImportError("install the interleaved-speech-data hubert dependencies") from exc
        extractor = AutoFeatureExtractor.from_pretrained(model, revision=revision)
        backend_normalize = bool(getattr(extractor, "do_normalize", False))
        self.model = HubertModel.from_pretrained(model, revision=revision).to(device).eval()
        self.normalize = backend_normalize if normalize is None else bool(normalize)
        kmeans_paths = [kmeans] if isinstance(kmeans, str) else list(kmeans)
        if not kmeans_paths:
            raise ValueError("at least one K-means model is required")
        self.centers = [
            torch.from_numpy(joblib.load(path).cluster_centers_).float().to(device)
            for path in kmeans_paths
        ]
        widths = {len(centers) for centers in self.centers}
        if len(widths) != 1:
            raise ValueError("all K-means codebooks must have the same vocabulary size")
        if next(iter(widths)) > np.iinfo(np.int16).max - 1:
            raise ValueError("K-means vocabulary plus EOS/PAD must fit in int16")
        self.center_norms = [centers.square().sum(1) for centers in self.centers]
        self.layer = layer
        self.device = device
        self.max_samples = 16000 * max_seconds
        self.eos_token = widths.pop()
        self.pad_token = self.eos_token + 1
        self.n_codebooks = len(self.centers)
        self.sample_rate = 16000

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "unit_extractor": "hubert-kmeans",
            "sample_rate": self.sample_rate,
            "collapsed": True,
            "alignment_timeline": "uncollapsed-hubert-to-collapsed-runs",
        }

    def _features(self, chunk: torch.Tensor) -> torch.Tensor:
        output = self.model(
            input_values=chunk.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
        if output.hidden_states is None or self.layer >= len(output.hidden_states):
            raise ValueError(
                f"mHuBERT layer {self.layer} is unavailable; "
                f"model returned {len(output.hidden_states or ())} hidden states"
            )
        return output.hidden_states[self.layer][0]

    @torch.inference_mode()
    def encode_uncollapsed(self, waveform: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(waveform, dtype=torch.float32, device=self.device)
        if self.normalize:
            x = F.layer_norm(x, x.shape)
        features = []
        for start in range(0, x.numel(), self.max_samples):
            chunk = x[start:start + self.max_samples]
            if chunk.numel() < 720:
                continue
            features.append(self._features(chunk))
        if not features:
            raise ValueError("audio is too short for HuBERT")
        feature = torch.cat(features)
        feature_norm = feature.square().sum(1, keepdim=True)
        codes = []
        for centers, center_norm in zip(self.centers, self.center_norms):
            distance = feature_norm - 2 * feature @ centers.T + center_norm
            codes.append(distance.argmin(1).cpu().numpy())
        return np.stack(codes, axis=1).astype(np.int16)

    @torch.inference_mode()
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        return collapse_codes(self.encode_uncollapsed(waveform), self.eos_token)

    @torch.inference_mode()
    def encode_with_mapping(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return collapse_codes_with_mapping(
            self.encode_uncollapsed(waveform), self.eos_token
        )


class MimiTokenizer:
    """Eight-stream, 12.5 Hz Mimi tokenizer used by Moshi pretraining."""

    def __init__(
        self,
        repo: str,
        revision: str,
        *,
        filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
        device: str = "cuda",
        n_codebooks: int = 8,
    ):
        if n_codebooks != 8:
            raise ValueError("the Moshi pretraining recipe requires eight Mimi codebooks")
        try:
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders
        except ImportError as exc:
            raise ImportError("Mimi tokenization requires the mimi dependencies") from exc
        weights = Path(
            hf_hub_download(repo_id=repo, filename=filename, revision=revision)
        )
        self.model = loaders.get_mimi(
            weights, device=device, num_codebooks=n_codebooks
        )
        self.device = device
        self.repo = repo
        self.revision = revision
        self.filename = filename
        self.weights_sha256 = _sha256(weights)
        self.sample_rate = int(loaders.SAMPLE_RATE)
        self.frame_rate = float(loaders.FRAME_RATE)
        self.n_codebooks = n_codebooks
        self.eos_token = 2048
        self.pad_token = 2048

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "unit_extractor": "mimi",
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "collapsed": False,
            "alignment_timeline": "mimi-frames",
            "mimi_repo": self.repo,
            "mimi_revision": self.revision,
            "mimi_filename": self.filename,
            "mimi_sha256": self.weights_sha256,
        }

    @torch.inference_mode()
    def encode(self, waveform: np.ndarray) -> np.ndarray:
        signal = torch.as_tensor(
            waveform, dtype=torch.float32, device=self.device
        ).reshape(1, 1, -1)
        codes = self.model.encode(signal)[0].transpose(0, 1).cpu().numpy()
        if codes.ndim != 2 or codes.shape[1] != self.n_codebooks:
            raise ValueError(f"Mimi returned invalid code shape {codes.shape}")
        if not len(codes) or codes.min() < 0 or codes.max() >= self.pad_token:
            raise ValueError("Mimi returned codes outside its 2,048-entry vocabulary")
        boundary = np.full((1, self.n_codebooks), self.pad_token, dtype=np.int16)
        return np.concatenate((codes.astype(np.int16), boundary), axis=0)

    @torch.inference_mode()
    def encode_with_mapping(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        codes = self.encode(waveform)
        return codes, np.arange(len(codes), dtype=np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_speech_tokenizer(
    kind: str,
    *,
    device: str,
    hubert_model: str | None = None,
    hubert_revision: str | None = None,
    kmeans: str | Sequence[str] | None = None,
    hubert_layer: int = 11,
    hubert_normalize: bool | None = None,
    mimi_repo: str | None = None,
    mimi_revision: str | None = None,
    mimi_filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
):
    if kind == "hubert":
        if not hubert_model or not hubert_revision or not kmeans:
            raise ValueError("HuBERT tokenization requires model, revision, and K-means")
        return HubertKMeansTokenizer(
            hubert_model,
            kmeans,
            layer=hubert_layer,
            device=device,
            revision=hubert_revision,
            normalize=hubert_normalize,
        )
    if kind == "mimi":
        if not mimi_repo or not mimi_revision:
            raise ValueError("Mimi tokenization requires repository and revision")
        return MimiTokenizer(
            mimi_repo,
            mimi_revision,
            filename=mimi_filename,
            device=device,
        )
    raise ValueError("speech tokenizer must be hubert or mimi")


class MMSForcedAligner:
    """Word timestamps from TorchAudio's multilingual MMS forced-aligner bundle."""

    def __init__(self, device: str = "cuda"):
        try:
            from torchaudio.pipelines import MMS_FA
        except (ImportError, AttributeError) as exc:
            raise ImportError("MMS alignment requires torchaudio with the MMS_FA bundle") from exc
        self.sample_rate = int(MMS_FA.sample_rate)
        self.model = MMS_FA.get_model().to(device).eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()
        self.device = device

    @staticmethod
    def normalize(text: str) -> list[str]:
        text = text.lower().replace("’", "'")
        text = re.sub(r"[^a-z' ]", " ", text)
        return re.sub(r" +", " ", text).strip().split()

    @torch.inference_mode()
    def align(self, waveform: np.ndarray, transcript: str) -> list[dict[str, float | str]]:
        words = self.normalize(transcript)
        if not words:
            raise ValueError("transcript is empty after MMS normalization")
        signal = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1).to(self.device)
        emission, _ = self.model(signal)
        spans = self.aligner(emission[0], self.tokenizer(words))
        if len(spans) != len(words) or any(not word_spans for word_spans in spans):
            raise ValueError("MMS returned an incomplete word alignment")
        seconds_per_frame = len(waveform) / self.sample_rate / emission.size(1)
        result = []
        previous_end = 0.0
        for word, word_spans in zip(words, spans):
            start = max(previous_end, float(word_spans[0].start) * seconds_per_frame)
            end = float(word_spans[-1].end) * seconds_per_frame
            if end <= start:
                raise ValueError(f"invalid MMS span for {word!r}")
            result.append({"text": word, "start": start, "end": end})
            previous_end = end
        return result


def align_words(
    words: list[dict[str, Any]],
    tokenizer,
    num_audio_steps: int,
    *,
    audio_seconds: float,
    pad_id: int,
    end_of_word_id: int,
    frame_to_code: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert timestamped words into token IDs plus run durations over audio steps."""
    if not words or audio_seconds <= 0:
        raise ValueError("word timestamps and positive audio duration are required")
    dense = np.full(num_audio_steps, pad_id, dtype=np.int32)
    previous_end = 0
    for word in words:
        text = str(word.get("text", "")).strip()
        start = word.get("start", word.get("start_ts"))
        end = word.get("end", word.get("end_ts"))
        if not text or start is None or end is None or end <= start:
            raise ValueError(f"invalid word timestamp: {word}")
        if float(start) < 0 or float(end) > audio_seconds + 1e-3:
            raise ValueError(f"word timestamp is outside the audio segment: {word}")
        if frame_to_code is None:
            lo = min(
                num_audio_steps - 1,
                round(float(start) / audio_seconds * num_audio_steps),
            )
            hi = min(
                num_audio_steps,
                max(lo + 1, round(float(end) / audio_seconds * num_audio_steps)),
            )
        else:
            frame_to_code = np.asarray(frame_to_code, dtype=np.int64)
            num_frames = len(frame_to_code) - 1
            if num_frames < 1 or frame_to_code[-1] != num_audio_steps - 1:
                raise ValueError("invalid uncollapsed-to-collapsed unit mapping")
            frame_lo = max(
                0,
                min(
                    num_frames - 1,
                    int(np.floor(float(start) / audio_seconds * num_frames)),
                ),
            )
            frame_hi = max(
                1,
                min(
                    num_frames,
                    int(np.ceil(float(end) / audio_seconds * num_frames)),
                ),
            )
            lo = int(frame_to_code[frame_lo])
            hi = min(num_audio_steps - 1, int(frame_to_code[frame_hi - 1]) + 1)
        lo = max(previous_end, lo)
        if lo >= num_audio_steps or hi <= lo:
            raise ValueError(f"word timestamp falls outside usable unit runs: {word}")
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"]) + [end_of_word_id]
        hi = min(num_audio_steps, max(hi, lo + len(ids)))
        if hi - lo < len(ids):
            raise ValueError(
                f"{text!r} has {len(ids)} text tokens but only {hi - lo} aligned unit steps"
            )
        boundaries = np.linspace(lo, hi, len(ids) + 1, dtype=np.int64)
        for token, a, b in zip(ids, boundaries[:-1], boundaries[1:]):
            if b > a:
                dense[a:b] = int(token)
        previous_end = hi
    changes = np.ones(num_audio_steps, dtype=bool)
    changes[1:] = dense[1:] != dense[:-1]
    starts = np.flatnonzero(changes)
    return dense[starts], np.diff(np.append(starts, num_audio_steps)).astype(np.int32)


def align_words_to_mimi(
    words: list[dict[str, Any]],
    tokenizer,
    num_audio_steps: int,
    *,
    audio_seconds: float,
    pad_id: int,
    end_of_word_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place wordpieces at word starts on Mimi's 12.5 Hz timeline."""
    num_frames = num_audio_steps - 1
    if not words or num_frames < 1 or audio_seconds <= 0:
        raise ValueError("word timestamps and Mimi frames are required")
    dense = np.full(num_audio_steps, pad_id, dtype=np.int32)
    starts = []
    for word in words:
        start = word.get("start", word.get("start_ts"))
        if start is None or not 0 <= float(start) < audio_seconds:
            raise ValueError(f"invalid word timestamp: {word}")
        starts.append(min(num_frames - 1, round(float(start) / audio_seconds * num_frames)))
    for index, word in enumerate(words):
        lo = starts[index]
        hi = starts[index + 1] if index + 1 < len(starts) else num_frames
        ids = list(
            tokenizer(str(word["text"]).strip(), add_special_tokens=False)["input_ids"]
        )
        if not ids or hi - lo < len(ids) + 1:
            raise ValueError(f"insufficient Mimi frames for aligned word: {word}")
        dense[lo : lo + len(ids)] = ids
        dense[hi - 1] = end_of_word_id
    changes = np.ones(num_audio_steps, dtype=bool)
    changes[1:] = dense[1:] != dense[:-1]
    positions = np.flatnonzero(changes)
    return dense[positions], np.diff(np.append(positions, num_audio_steps)).astype(np.int32)
