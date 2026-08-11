import os
import torch
import json
import random
import hashlib
import numpy as np
from pathlib import Path
from numba import njit
from typing import List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


@dataclass
class DatasetArgs:
    data_root: str = "data"
    txt_datasets: List[str] = field(default_factory=list)
    img_datasets: List[str] = field(default_factory=list)
    aud_datasets: List[str] = field(default_factory=list)
    imgtxt_datasets: List[str] = field(default_factory=list)
    audtxt_datasets: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=list)
    p_strategies: List[float] = field(default_factory=list)
    txt_tokens: str = ""
    img_tokens: str = ""
    aud_tokens: str = ""
    audtxt_mode: str = "interleaved"
    block_size: int = 2048
    txt_datasets_probs: Optional[List[float]] = None
    img_datasets_probs: Optional[List[float]] = None
    aud_datasets_probs: Optional[List[float]] = None
    imgtxt_datasets_probs: Optional[List[float]] = None
    audtxt_datasets_probs: Optional[List[float]] = None

MIN_WORDS_TXT_CHUNK = 10
MAX_WORDS_TXT_CHUNK = 31
MIN_WORDS_OTHER_CHUNK = 5
MAX_WORDS_OTHER_CHUNK = 16
P_FIRST_TXT_INTERLEAVED = 0.5


@njit(nogil=True, cache=True)
def _seed_numba(seed):
    np.random.seed(seed)

@njit(nogil=True, cache=True)
def _build_interleaved_sample(
    sample_len,
    other_nc,
    other_pad,
    txt_pad,
    txt_epad,
    txt_swt_token,
    lens, bnds,
    flat_txt, flat_other
):
    rng = np.random.random
    randint = np.random.randint

    res = np.full((sample_len, other_nc + 1), other_pad, np.int64)
    res[:, 0] = txt_pad

    txt_in   = np.zeros(sample_len, np.bool_)
    other_in = np.zeros(sample_len, np.bool_)
    txt_prd  = np.zeros(sample_len, np.bool_)
    other_prd= np.zeros(sample_len, np.bool_)

    cur = 0
    wrd_cnt = 0
    is_txt = rng() < P_FIRST_TXT_INTERLEAVED
    chunk_len = randint(MIN_WORDS_TXT_CHUNK, MAX_WORDS_TXT_CHUNK) if is_txt else randint(MIN_WORDS_OTHER_CHUNK, MAX_WORDS_OTHER_CHUNK)

    while True:
        ix = randint(0, lens.size)
        seg_len = lens[ix]
        lo = bnds[ix]

        seg_txt = flat_txt[lo: lo + seg_len]
        seg_oth = flat_other[lo: lo + seg_len]

        for i in range(seg_len):

            if wrd_cnt == chunk_len:
                if cur == sample_len:
                    return res, txt_in, other_in, txt_prd, other_prd

                res[cur, 0] = txt_swt_token
                txt_in[cur] = True
                cur += 1

                wrd_cnt = 0
                is_txt = not is_txt
                chunk_len = randint(MIN_WORDS_TXT_CHUNK, MAX_WORDS_TXT_CHUNK) if is_txt else randint(MIN_WORDS_OTHER_CHUNK, MAX_WORDS_OTHER_CHUNK)

                if cur == sample_len:
                    return res, txt_in, other_in, txt_prd, other_prd

            if cur == sample_len:
                return res, txt_in, other_in, txt_prd, other_prd

            t = seg_txt[i]

            if is_txt:
                if t != txt_pad and t != txt_epad:
                    res[cur, 0] = t
                    txt_in[cur] = True
                    txt_prd[cur] = True
                    cur += 1
                elif t == txt_epad:
                    wrd_cnt += 1

            else:
                res[cur, 1:] = seg_oth[i]
                other_in[cur] = True
                other_prd[cur] = True
                cur += 1

                if t == txt_epad:
                    wrd_cnt += 1


@njit(nogil=True, cache=True)
def _build_interleaved_sample_compact(
    sample_len,
    other_nc,
    other_pad,
    txt_pad,
    txt_epad,
    txt_swt_token,
    audio_lens,
    audio_bnds,
    text_lens,
    text_bnds,
    flat_text,
    flat_durations,
    flat_other,
):
    res = np.full((sample_len, other_nc + 1), other_pad, np.int64)
    res[:, 0] = txt_pad
    txt_in = np.zeros(sample_len, np.bool_)
    other_in = np.zeros(sample_len, np.bool_)
    txt_prd = np.zeros(sample_len, np.bool_)
    other_prd = np.zeros(sample_len, np.bool_)
    cur = 0
    word_count = 0
    is_text = np.random.random() < P_FIRST_TXT_INTERLEAVED
    chunk_words = np.random.randint(MIN_WORDS_TXT_CHUNK, MAX_WORDS_TXT_CHUNK) if is_text else np.random.randint(MIN_WORDS_OTHER_CHUNK, MAX_WORDS_OTHER_CHUNK)

    while cur < sample_len:
        sample = np.random.randint(0, audio_lens.size)
        audio_pos = audio_bnds[sample]
        text_lo = text_bnds[sample]
        text_hi = text_bnds[sample + 1]
        for text_pos in range(text_lo, text_hi):
            if word_count == chunk_words:
                res[cur, 0] = txt_swt_token
                txt_in[cur] = True
                cur += 1
                if cur == sample_len:
                    return res, txt_in, other_in, txt_prd, other_prd
                word_count = 0
                is_text = not is_text
                chunk_words = np.random.randint(MIN_WORDS_TXT_CHUNK, MAX_WORDS_TXT_CHUNK) if is_text else np.random.randint(MIN_WORDS_OTHER_CHUNK, MAX_WORDS_OTHER_CHUNK)

            token = flat_text[text_pos]
            duration = flat_durations[text_pos]
            if is_text:
                if token != txt_pad and token != txt_epad:
                    res[cur, 0] = token
                    txt_in[cur] = True
                    txt_prd[cur] = True
                    cur += 1
            else:
                for offset in range(duration):
                    if cur == sample_len:
                        return res, txt_in, other_in, txt_prd, other_prd
                    res[cur, 1:] = flat_other[audio_pos + offset]
                    other_in[cur] = True
                    other_prd[cur] = True
                    cur += 1
            audio_pos += duration
            if token == txt_epad:
                word_count += 1
            if cur == sample_len:
                return res, txt_in, other_in, txt_prd, other_prd
    return res, txt_in, other_in, txt_prd, other_prd


@njit(nogil=True, cache=True)
def _build_aligned_sample_compact(
    sample_len,
    audio_nc,
    audio_pad,
    text_pad,
    audio_lens,
    audio_bnds,
    text_lens,
    text_bnds,
    flat_text,
    flat_durations,
    flat_audio,
):
    tokens = np.full((sample_len, audio_nc + 1), audio_pad, np.int64)
    tokens[:, 0] = text_pad
    text_in = np.zeros(sample_len, np.bool_)
    audio_in = np.ones(sample_len, np.bool_)
    text_pred = np.ones(sample_len, np.bool_)
    audio_pred = np.ones(sample_len, np.bool_)
    cur = 0
    while cur < sample_len:
        sample = np.random.randint(0, audio_lens.size)
        audio_pos = audio_bnds[sample]
        text_lo = text_bnds[sample]
        text_hi = text_bnds[sample + 1]
        for text_pos in range(text_lo, text_hi):
            token = flat_text[text_pos]
            duration = flat_durations[text_pos]
            for offset in range(duration):
                if cur == sample_len:
                    return tokens, text_in, audio_in, text_pred, audio_pred
                tokens[cur, 0] = token
                tokens[cur, 1:] = flat_audio[audio_pos + offset]
                text_in[cur] = token != text_pad
                cur += 1
            audio_pos += duration
    return tokens, text_in, audio_in, text_pred, audio_pred


def _tokenizer_base_name(tok_id):
    if "/" in tok_id:
        org, tail = tok_id.split("/", 1)
    else:
        org, tail = "", tok_id
    base = tail.rsplit("-", 1)[0]  # "SmolLM-1.7B" -> "SmolLM"
    return org, base


def _load_and_check_txt_meta(config, has_txt_unimodal, has_imgtxt, has_audtxt, txt_alias):
    """
    Load meta_txt.json for all text-involving datasets and check consistency
    across them. Allows tokenizer_id variants like 1.7B vs 135M, but enforces
    equality for tokenizer_alias, base_vocab_size, eos_id and (if present)
    pad_id / epad_id / special_vocab_size.

    Returns a dict with the merged canonical metadata.
    """
    meta_specs = []  # (meta_path, source_str)

    if has_txt_unimodal:
        for ds in config.txt_datasets:
            meta_specs.append(
                (os.path.join(config.data_root, ds, config.txt_tokens, "meta_txt.json"),
                 f"txt:{ds}")
            )

    if has_imgtxt:
        for ds in config.imgtxt_datasets:
            meta_specs.append(
                (os.path.join(config.data_root, ds, config.img_tokens, "meta_txt.json"),
                 f"imgtxt:{ds}")
            )

    if has_audtxt:
        for ds in config.audtxt_datasets:
            meta_specs.append(
                (os.path.join(config.data_root, ds, config.aud_tokens, "meta_txt.json"),
                 f"audtxt:{ds}")
            )

    assert meta_specs, "Text-related tasks selected but no text datasets provided"

    merged_meta = None

    for meta_path, src in meta_specs:
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                raw_meta = json.load(f)
        else:
            token_root = os.path.dirname(meta_path)
            candidates = sorted(Path(token_root).glob("*/*/meta.json"))
            if not candidates:
                raise FileNotFoundError(meta_path)
            packed = json.loads(candidates[0].read_text())["preprocessing"]
            raw_meta = {
                "tokenizer_id": packed["text_tokenizer"],
                "tokenizer_alias": txt_alias,
                "base_vocab_size": packed["text_base_vocab_size"],
                "eos_id": packed["text_eos_id"],
                "pad_id": packed["text_pad_id"],
                "epad_id": packed["text_end_of_word_id"],
                "special_vocab_size": packed["text_special_vocab_size"],
            }

        meta = raw_meta[txt_alias] if isinstance(raw_meta, dict) and txt_alias in raw_meta else raw_meta

        try:
            tokenizer_id = meta["tokenizer_id"]
            tokenizer_alias = meta.get("tokenizer_alias", txt_alias)
            base_vocab_size = meta["base_vocab_size"]
            eos_id = meta["eos_id"]
        except KeyError as e:
            raise KeyError(f"Missing required key {e} in {meta_path} for {src}")

        pad_id = meta.get("pad_id")
        epad_id = meta.get("epad_id")
        special_vocab_size = meta.get("special_vocab_size")

        current = {
            "tokenizer_id": tokenizer_id,
            "tokenizer_alias": tokenizer_alias,
            "base_vocab_size": base_vocab_size,
            "pad_id": pad_id,
            "epad_id": epad_id,
            "eos_id": eos_id,
            "special_vocab_size": special_vocab_size,
            "source": src,
        }

        if merged_meta is None:
            merged_meta = current
            continue

        ref = merged_meta

        if current["tokenizer_id"] != ref["tokenizer_id"]:
            org_ref, base_ref = _tokenizer_base_name(ref["tokenizer_id"])
            org_cur, base_cur = _tokenizer_base_name(current["tokenizer_id"])
            if org_ref == org_cur and base_ref == base_cur:
                print(
                    f"[MultimodalPretrainDataset] Warning: tokenizer_id mismatch "
                    f"between datasets '{ref['source']}' and '{current['source']}': "
                    f"'{ref['tokenizer_id']}' vs '{current['tokenizer_id']}'. "
                    f"Assuming they share the same tokenizer."
                )
            else:
                raise ValueError(
                    "Incompatible tokenizer_id between datasets "
                    f"'{ref['source']}' ({ref['tokenizer_id']}) and "
                    f"'{current['source']}' ({current['tokenizer_id']})."
                )

        if (
            current["tokenizer_alias"] is not None
            and ref["tokenizer_alias"] is not None
            and current["tokenizer_alias"] != ref["tokenizer_alias"]
        ):
            raise ValueError(
                "tokenizer_alias mismatch between datasets "
                f"'{ref['source']}' ({ref['tokenizer_alias']}) and "
                f"'{current['source']}' ({current['tokenizer_alias']})."
            )

        if current["base_vocab_size"] != ref["base_vocab_size"]:
            raise ValueError(
                "base_vocab_size mismatch between datasets "
                f"'{ref['source']}' ({ref['base_vocab_size']}) and "
                f"'{current['source']}' ({current['base_vocab_size']})."
            )

        if current["eos_id"] != ref["eos_id"]:
            raise ValueError(
                "eos_id mismatch between datasets "
                f"'{ref['source']}' ({ref['eos_id']}) and "
                f"'{current['source']}' ({current['eos_id']})."
            )

        for key in ("pad_id", "epad_id", "special_vocab_size"):
            if current[key] is not None and ref[key] is not None and current[key] != ref[key]:
                raise ValueError(
                    f"{key} mismatch between datasets "
                    f"'{ref['source']}' ({ref[key]}) and "
                    f"'{current['source']}' ({current[key]})."
                )
            if ref[key] is None and current[key] is not None:
                ref[key] = current[key]

    assert merged_meta is not None, "Failed to load any text metadata"
    return merged_meta



class MultimodalPretrainDataset(IterableDataset):
    def __init__(self, config: DatasetArgs):
        super().__init__()
        self.config = config

        if len(config.p_strategies) != 5:
            raise ValueError("p_strategies must contain txt, img, aud, imgtxt, and audtxt weights")
        if config.audtxt_mode not in {"interleaved", "aligned"}:
            raise ValueError("audtxt_mode must be interleaved or aligned")
        if config.audtxt_mode == "aligned" and config.p_strategies[3]:
            raise ValueError("aligned audio-text mode cannot be mixed with image-text interleaving")
        if not all(np.isfinite(value) and value >= 0 for value in config.p_strategies):
            raise ValueError("p_strategies must contain finite non-negative weights")
        if not any(config.p_strategies):
            raise ValueError("at least one training strategy must have positive weight")
        if not config.splits:
            raise ValueError("at least one data split is required")
        modalities = ("txt", "img", "aud", "imgtxt", "audtxt")
        for enabled, modality in zip(config.p_strategies, modalities):
            datasets = getattr(config, f"{modality}_datasets")
            probabilities = getattr(config, f"{modality}_datasets_probs")
            if enabled and not datasets:
                raise ValueError(f"{modality} is enabled but has no datasets")
            if probabilities is not None:
                if len(probabilities) != len(datasets):
                    raise ValueError(f"{modality}_datasets_probs must match its dataset list")
                if not probabilities or not all(
                    np.isfinite(value) and value >= 0 for value in probabilities
                ) or not any(probabilities):
                    raise ValueError(
                        f"{modality}_datasets_probs must contain finite non-negative weights"
                    )

        sampling_strategies = [
            ("get_unimodal_sample", "txt"),
            ("get_unimodal_sample", "img"),
            ("get_unimodal_sample", "aud"),
            ("get_interleaved_sample", "imgtxt"),
            ("get_interleaved_sample", "audtxt"),
        ]
        self.samplers, self.sampler_probs = [], []
        for p, sampler in zip(config.p_strategies, sampling_strategies):
            if p:
                self.samplers.append(sampler)
                self.sampler_probs.append(p)

        self.token_ids = {
            "txt": config.txt_tokens,
            "img": config.img_tokens,
            "aud": config.aud_tokens,
        }

        self.txt_pad_token = None
        self.txt_epad_token = None
        self.txt_swt_token = None
        self.txt_vocabsize = None
        self.txt_eos_token = None
        self.txt_tokenizer_id = None
        self.txt_tokenizer_alias = None
        self.txt_base_vocab_size = None

        self.img_vocabsize = 0
        self.img_pad_token = None
        self.img_eos_token = None
        self.img_ncodebooks = 0

        self.aud_vocabsize = 0
        self.aud_pad_token = None
        self.aud_eos_token = None
        self.aud_ncodebooks = 0

        has_txt_unimodal = bool(config.p_strategies[0])
        has_img_unimodal = bool(config.p_strategies[1])
        has_aud_unimodal = bool(config.p_strategies[2])
        has_imgtxt       = bool(config.p_strategies[3])
        has_audtxt       = bool(config.p_strategies[4])

        has_any_text_task = has_txt_unimodal or has_imgtxt or has_audtxt

        if has_any_text_task:
            txt_alias = self.config.txt_tokens  # e.g. "smollm"
            ref = _load_and_check_txt_meta(
                config=config,
                has_txt_unimodal=has_txt_unimodal,
                has_imgtxt=has_imgtxt,
                has_audtxt=has_audtxt,
                txt_alias=txt_alias,
            )

            self.txt_tokenizer_id = ref["tokenizer_id"]
            self.txt_tokenizer_alias = ref["tokenizer_alias"]
            self.txt_base_vocab_size = ref["base_vocab_size"]
            self.txt_eos_token = ref["eos_id"]

            base_vocab_size = self.txt_base_vocab_size
            pad_id = ref["pad_id"]
            epad_id = ref["epad_id"]
            special_vocab_size = ref["special_vocab_size"]

            has_interleaved = has_imgtxt or has_audtxt
            only_txt_unimodal = (
                has_txt_unimodal
                and not has_interleaved
                and not has_img_unimodal
                and not has_aud_unimodal
            )
            multi_unimodal = (
                has_txt_unimodal
                and not has_interleaved
                and (has_img_unimodal or has_aud_unimodal)
            )

            if has_interleaved:
                if pad_id is None or epad_id is None or special_vocab_size is None:
                    raise ValueError(
                        "Interleaved text tasks require pad_id, epad_id and special_vocab_size "
                        "to be defined in all text-related metadata."
                    )
                expected_special = base_vocab_size + 2  # pad + epad
                if special_vocab_size != expected_special:
                    raise ValueError(
                        f"special_vocab_size ({special_vocab_size}) does not match "
                        f"base_vocab_size + 2 ({expected_special}) for interleaved tasks."
                    )

                if pad_id == epad_id:
                    raise ValueError("pad_id and epad_id must be distinct")
                if not (0 <= pad_id < special_vocab_size and 0 <= epad_id < special_vocab_size):
                    raise ValueError(
                        "pad_id and epad_id must be contained in special_vocab_size"
                    )

                self.txt_pad_token = pad_id
                self.txt_epad_token = epad_id
                if has_audtxt and config.audtxt_mode == "aligned":
                    self.txt_vocabsize = special_vocab_size
                else:
                    self.txt_swt_token = special_vocab_size
                    self.txt_vocabsize = special_vocab_size + 1

            elif multi_unimodal:
                if pad_id is None:
                    raise ValueError(
                        "Multiple unimodal tasks (including text) require pad_id "
                        "to be defined in text metadata."
                    )
                self.txt_pad_token = pad_id
                self.txt_vocabsize = base_vocab_size + 1

            elif only_txt_unimodal:
                self.txt_vocabsize = base_vocab_size
            else:
                self.txt_vocabsize = base_vocab_size

        img_aud_specs = [
            ("img", 1, 3, "img_datasets", "imgtxt_datasets", "img_tokens", "meta_img.json"),
            ("aud", 2, 4, "aud_datasets", "audtxt_datasets", "aud_tokens", "meta_aud.json"),
        ]

        for kind, base_i, pair_i, base_attr, pair_attr, tokens_attr, meta_file in img_aud_specs:
            if not (config.p_strategies[base_i] or config.p_strategies[pair_i]):
                continue

            ds_attr = pair_attr if config.p_strategies[pair_i] else base_attr

            meta_path = os.path.join(
                config.data_root,
                getattr(config, ds_attr)[0],
                getattr(self.config, tokens_attr),
                meta_file,  # <- always meta_{img,aud}.json now
            )
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                token_root = os.path.dirname(meta_path)
                candidates = sorted(Path(token_root).glob("*/*/meta.json"))
                if not candidates:
                    raise FileNotFoundError(meta_path)
                packed = json.loads(candidates[0].read_text())
                if packed.get("modality") == "rendered-text":
                    meta = {
                        "vocab_size": packed["image_vocab_size"],
                        "pad_id": packed["image_pad_id"],
                        "eos_id": packed["image_eos_id"],
                        "n_q": packed["n_codebooks"],
                    }
                else:
                    prep = packed["preprocessing"]
                    meta = {
                        "vocab_size": prep["vocab_size"],
                        "pad_id": prep["pad_token"],
                        "eos_id": prep["eos_token"],
                        "n_q": packed["n_codebooks"],
                    }

            setattr(self, f"{kind}_vocabsize",  meta["vocab_size"])
            setattr(self, f"{kind}_pad_token",  meta["pad_id"])
            setattr(self, f"{kind}_eos_token",  meta["eos_id"])
            setattr(self, f"{kind}_ncodebooks", meta["n_q"])

        txt_ncodebooks = 1 if self.txt_vocabsize is not None else 0
        self.n_codebooks = txt_ncodebooks + self.img_ncodebooks + self.aud_ncodebooks

        start = txt_ncodebooks
        self.img_slice = slice(start, start + self.img_ncodebooks)
        self.aud_slice = slice(start + self.img_ncodebooks,
                               start + self.img_ncodebooks + self.aud_ncodebooks)

        self.sample_len = config.block_size + 1
        modalities = ["txt", "img", "aud", "imgtxt", "audtxt"]
        for i, m in enumerate(modalities):
            if not config.p_strategies[i]:
                continue
            print(f"Loading {m} data ...")
            self._load_modality(
                modality=m,
                datasets=getattr(config, f"{m}_datasets"),
                splits=config.splits,
                shards_attr=f"{m}_shards",
                probs_attr=f"{m}_shards_probs",
                datasets_probs=getattr(config, f"{m}_datasets_probs"),
            )

        self.iteration_seed = 1337
        self.start_index = 0
        self.rank = 0

    def set_iteration(self, *, seed: int, start_index: int = 0, rank: int = 0):
        if start_index < 0 or rank < 0:
            raise ValueError("start_index and rank must be non-negative")
        self.iteration_seed = int(seed)
        self.start_index = int(start_index)
        self.rank = int(rank)

    def fingerprint(self) -> str:
        """Hash the data configuration plus canonical shard metadata."""
        digest = hashlib.sha256(json.dumps(self.config.__dict__, sort_keys=True).encode())
        paths = set()
        modalities = ("txt", "img", "aud", "imgtxt", "audtxt")
        for enabled, modality in zip(self.config.p_strategies, modalities):
            if not enabled:
                continue
            token_kind = modality.replace("txt", "") if modality.endswith("txt") else modality
            token_name = getattr(self.config, f"{token_kind}_tokens")
            for dataset in getattr(self.config, f"{modality}_datasets"):
                root = Path(self.config.data_root) / dataset / token_name
                paths.update(root.glob("meta_*.json"))
                for split in self.config.splits:
                    paths.update((root / split).glob("*/meta.json"))
        for path in sorted(paths):
            digest.update(str(path.relative_to(self.config.data_root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _load_modality(
        self,
        modality,
        datasets,
        splits,
        shards_attr,
        probs_attr,
        datasets_probs=None,
    ):
        def iter_shards():
            for ds in datasets:
                for sp in splits:
                    data = self.get_data_mmaps(sp, ds, modality)
                    if data is None:
                        continue
                    for shard in data if isinstance(data, list) else [data]:
                        if modality in ("imgtxt", "audtxt") and "txt" not in shard:
                            continue
                        yield ds, shard

        dataset_to_shards = defaultdict(list)
        for ds, shard in iter_shards():
            key = modality.replace("txt", "") if modality.endswith("txt") else modality
            if modality in ("txt", "img", "aud") and len(shard[key]) < self.sample_len:
                continue
            dataset_to_shards[ds].append(shard)

        if not dataset_to_shards:
            raise ValueError(
                f"no {modality} data found for datasets {datasets}, splits {splits}"
            )

        out_shards, out_probs = [], []

        key = modality.replace('txt', '') if modality in ['imgtxt', 'audtxt'] else modality
        if datasets_probs:
            weights = dict(zip(datasets, datasets_probs))
            for ds, shards in dataset_to_shards.items():
                ds_lens = [len(s[key]) for s in shards]
                ds_size = sum(ds_lens)
                w = weights[ds]
                out_shards.extend(shards)
                out_probs.extend([w * shard_len / ds_size for shard_len in ds_lens])
        else:
            out_shards = [s for shards in dataset_to_shards.values() for s in shards]
            lens = [len(s[key]) for s in out_shards]
            total = sum(lens)
            out_probs = [shard_len / total for shard_len in lens]

        setattr(self, shards_attr, out_shards)
        setattr(self, probs_attr, out_probs)
            
    def get_data_mmaps(self, split, dataset, modality):
        key = modality.replace("txt", "") if modality in ("imgtxt", "audtxt") else modality
        tokens_source = self.token_ids[modality if modality in ['txt', 'img', 'aud'] else key]
        data_path = Path(self.config.data_root) / dataset / tokens_source / split
        if not data_path.is_dir():
            return None
        packed = []
        for shard_path in sorted(data_path.glob("*/meta.json")):
            shard_dir = shard_path.parent
            meta = json.loads(shard_path.read_text())
            if meta.get("schema_version") != 1:
                raise ValueError(f"unsupported canonical schema in {shard_dir}")
            if meta.get("modality") == "text":
                if modality != "txt":
                    continue
                lengths = np.memmap(shard_dir / "text.len", dtype=np.int64, mode="r")
                boundaries = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
                if int(boundaries[-1]) != int(meta["num_tokens"]):
                    raise ValueError(f"text token count mismatch in {shard_dir}")
                text = np.memmap(shard_dir / "text.bin", dtype=np.int32, mode="r")
                if len(text) != int(boundaries[-1]):
                    raise ValueError(f"text data size mismatch in {shard_dir}")
                if (
                    int(meta["base_vocab_size"]) != self.txt_base_vocab_size
                    or int(meta["eos_id"]) != self.txt_eos_token
                ):
                    raise ValueError(f"text vocabulary metadata mismatch in {shard_dir}")
                packed.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "txt": text,
                        "bnds": boundaries,
                        "lens": lengths,
                        "txt_conditioned": False,
                    }
                )
                continue
            if meta.get("modality") == "rendered-text":
                if key != "img":
                    continue
                lengths = np.memmap(shard_dir / "image.len", dtype=np.int64, mode="r")
                boundaries = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
                n_codebooks = int(meta["n_codebooks"])
                if (
                    n_codebooks != self.img_ncodebooks
                    or int(meta["image_vocab_size"]) != self.img_vocabsize
                    or int(meta["image_pad_id"]) != self.img_pad_token
                    or int(meta["image_eos_id"]) != self.img_eos_token
                ):
                    raise ValueError(f"image codebook mismatch in {shard_dir}")
                image = np.memmap(
                    shard_dir / "image.bin",
                    dtype=np.int16,
                    mode="r",
                    shape=(int(boundaries[-1]), n_codebooks),
                )
                item = {
                    "dataset": dataset,
                    "split": split,
                    "img": image,
                    "bnds": boundaries,
                    "lens": lengths,
                    "txt_conditioned": False,
                }
                if modality == "imgtxt":
                    text = np.memmap(shard_dir / "text.bin", dtype=np.int32, mode="r")
                    if len(text) != len(image):
                        raise ValueError(f"rendered-text alignment length mismatch in {shard_dir}")
                    item["txt"] = text
                packed.append(item)
                continue
            if key != "aud":
                continue
            lengths = np.memmap(shard_dir / "audio.len", dtype=np.int64, mode="r")
            boundaries = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
            n_codebooks = int(meta["n_codebooks"])
            preprocessing = meta["preprocessing"]
            if (
                n_codebooks != self.aud_ncodebooks
                or int(preprocessing["vocab_size"]) != self.aud_vocabsize
                or int(preprocessing["pad_token"]) != self.aud_pad_token
                or int(preprocessing["eos_token"]) != self.aud_eos_token
            ):
                raise ValueError(f"audio codebook mismatch in {shard_dir}")
            audio = np.memmap(
                shard_dir / "audio.bin",
                dtype=np.int16,
                mode="r",
                shape=(int(boundaries[-1]), n_codebooks),
            )
            item = {
                "dataset": dataset,
                "split": split,
                "aud": audio,
                "bnds": boundaries,
                "lens": lengths,
                "txt_conditioned": False,
            }
            if modality == "audtxt":
                if meta.get("profile") != "aligned":
                    raise ValueError(f"{shard_dir} is units-only but aligned data is required")
                text_lengths = np.memmap(shard_dir / "text.len", dtype=np.int64, mode="r")
                text_boundaries = np.concatenate(
                    ([0], np.cumsum(text_lengths, dtype=np.int64))
                )
                text = np.memmap(shard_dir / "text.bin", dtype=np.int32, mode="r")
                durations = np.memmap(shard_dir / "dur.bin", dtype=np.int32, mode="r")
                if len(text) != int(text_boundaries[-1]) or len(durations) != len(text):
                    raise ValueError(f"speech alignment size mismatch in {shard_dir}")
                item.update(
                    txt=text,
                    txt_durations=durations,
                    txt_lens=text_lengths,
                    txt_bnds=text_boundaries,
                    canonical_aligned=True,
                )
            packed.append(item)
        return packed or None

    def get_unimodal_sample(self, data):
        key = "txt" if "txt" in data else ("img" if "img" in data else "aud")
        S = self.sample_len
        max_start = len(data[key]) - S
        if max_start < 0:
            raise ValueError(
                f"{data['dataset']}/{data['split']} has {len(data[key])} {key} tokens; "
                f"at least {S} are required"
            )
        ix = random.randint(0, max_start)
        if self.txt_vocabsize is not None:
            txt = np.full((S, 1), self.txt_pad_token if self.txt_pad_token is not None else self.txt_eos_token, dtype=np.int64)  # unimodal text doesn't use pad; initialize with EOS
        if self.img_ncodebooks:
            img = np.full((S, self.img_ncodebooks), self.img_pad_token, np.int64)
        if self.aud_ncodebooks:
            aud = np.full((S, self.aud_ncodebooks), self.aud_pad_token, np.int64)
        txt_in = np.zeros(S, bool)
        img_in = np.zeros(S, bool)
        aud_in = np.zeros(S, bool)
        txt_pred = np.zeros(S, bool)
        img_pred = np.zeros(S, bool)
        aud_pred = np.zeros(S, bool)
        if key == "txt":
            txt[:, 0] = data["txt"][ix:ix + S]
            txt_in[:] = True
            txt_pred[:] = True
        elif key == "img":
            img[:] = data["img"][ix:ix + S]
            img_in[:] = True
            img_pred[:] = True
        else:  # "aud"
            aud[:] = data["aud"][ix:ix + S]
            aud_in[:] = True
            aud_pred[:] = True
        cols = []
        if self.txt_vocabsize is not None:
            cols.append(txt)  # (S,1)
        if self.img_ncodebooks:
            cols.append(img)  # (S,img_ncodebooks)
        if self.aud_ncodebooks:
            cols.append(aud)  # (S,aud_ncodebooks)
        sample = np.concatenate(cols, axis=1)  # (S, C = self.n_codebooks)
        return sample, txt_in, img_in, aud_in, txt_pred, img_pred, aud_pred

    def get_interleaved_sample(self, data, kind):
        if kind == "imgtxt":
            other_key = "img"
            other_nc  = self.img_ncodebooks
            other_pad = self.img_pad_token
            other_slice = self.img_slice
        else:  # "audtxt"
            other_key = "aud"
            other_nc  = self.aud_ncodebooks
            other_pad = self.aud_pad_token
            other_slice = self.aud_slice

        if data.get("canonical_aligned"):
            small, txt_in, other_in, txt_prd, other_prd = _build_interleaved_sample_compact(
                self.sample_len,
                other_nc,
                other_pad,
                self.txt_pad_token,
                self.txt_epad_token,
                self.txt_swt_token,
                data["lens"],
                data["bnds"],
                data["txt_lens"],
                data["txt_bnds"],
                data["txt"],
                data["txt_durations"],
                data[other_key],
            )
        else:
            small, txt_in, other_in, txt_prd, other_prd = _build_interleaved_sample(
                self.sample_len,
                other_nc,
                other_pad,
                self.txt_pad_token,
                self.txt_epad_token,
                self.txt_swt_token,
                data["lens"], data["bnds"],
                data["txt"], data[other_key],
            )

        S = self.sample_len
        full = np.full((S, self.n_codebooks), other_pad, np.int64)
        full[:, 0] = self.txt_pad_token
        if self.img_ncodebooks:
            full[:, self.img_slice] = self.img_pad_token
        if self.aud_ncodebooks:
            full[:, self.aud_slice] = self.aud_pad_token

        full[:, 0] = small[:, 0]
        full[:, other_slice] = small[:, 1:]

        txt_input_mask = txt_in
        img_input_mask = np.zeros(S, bool)
        aud_input_mask = np.zeros(S, bool)
        txt_preds_mask = txt_prd
        img_preds_mask = np.zeros(S, bool)
        aud_preds_mask = np.zeros(S, bool)

        if kind == "imgtxt":
            img_input_mask = other_in
            img_preds_mask = other_prd
        else:
            aud_input_mask = other_in
            aud_preds_mask = other_prd

        return (
            full[:, :self.n_codebooks],
            txt_input_mask, img_input_mask, aud_input_mask,
            txt_preds_mask, img_preds_mask, aud_preds_mask,
        )

    def get_aligned_audio_text_sample(self, data):
        if not data.get("canonical_aligned"):
            raise ValueError("aligned audio-text sampling requires canonical aligned shards")
        tokens, txt_in, aud_in, txt_pred, aud_pred = _build_aligned_sample_compact(
            self.sample_len,
            self.aud_ncodebooks,
            self.aud_pad_token,
            self.txt_pad_token,
            data["lens"],
            data["bnds"],
            data["txt_lens"],
            data["txt_bnds"],
            data["txt"],
            data["txt_durations"],
            data["aud"],
        )
        empty = np.zeros(self.sample_len, bool)
        return tokens, txt_in, empty, aud_in, txt_pred, empty.copy(), aud_pred

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        sample_index = self.start_index + worker_id
        shard_map = {
            "txt":    ("txt_shards",    "txt_shards_probs"),
            "img":    ("img_shards",    "img_shards_probs"),
            "aud":    ("aud_shards",    "aud_shards_probs"),
            "imgtxt": ("imgtxt_shards", "imgtxt_shards_probs"),
            "audtxt": ("audtxt_shards", "audtxt_shards_probs"),
        }

        while True:
            sample_seed = (
                self.iteration_seed
                + 1_000_003 * self.rank
                + 97_409 * sample_index
            ) % 2**32
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            _seed_numba(sample_seed)
            sampler, sampler_label = random.choices(
                self.samplers, weights=self.sampler_probs, k=1
            )[0]

            shards_attr, probs_attr = shard_map[sampler_label]
            data = random.choices(
                getattr(self, shards_attr),
                weights=getattr(self, probs_attr),
                k=1
            )[0]

            self.last_choice = {
                "sampler_label": sampler_label,
                "dataset": data["dataset"],
                "split": data["split"],
            }

            if sampler_label == "audtxt" and self.config.audtxt_mode == "aligned":
                out = self.get_aligned_audio_text_sample(data)
            elif sampler == "get_interleaved_sample":
                out = getattr(self, sampler)(data, sampler_label)
            else:
                out = getattr(self, sampler)(data)

            (tokens,
            txt_in, img_in, aud_in,
            txt_prd, img_prd, aud_prd) = out

            yield (
                torch.from_numpy(tokens.astype(np.int64)),
                torch.from_numpy(txt_in),
                torch.from_numpy(img_in),
                torch.from_numpy(aud_in),
                torch.from_numpy(txt_prd),
                torch.from_numpy(img_prd),
                torch.from_numpy(aud_prd),
            )
            sample_index += num_workers

def collate_fn(batch):
    (tokens,
     txt_in, img_in, aud_in,
     txt_prd, img_prd, aud_prd) = zip(*batch)

    tokens  = torch.stack(tokens)
    txt_in  = torch.stack(txt_in)
    img_in  = torch.stack(img_in)
    aud_in  = torch.stack(aud_in)

    txt_prd = torch.stack(txt_prd)[..., 1:]
    img_prd = torch.stack(img_prd)[..., 1:]
    aud_prd = torch.stack(aud_prd)[..., 1:]

    return tokens, txt_in, img_in, aud_in, txt_prd, img_prd, aud_prd

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers, dataset):
        worker_generator = torch.Generator()
        worker_generator.manual_seed(dataset.iteration_seed + dataset.rank)
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=num_workers > 0,
            generator=worker_generator,
        )

        for tokens, txt_in, img_in, aud_in, txt_prd, img_prd, aud_prd in dl:
            if str(device).startswith("cuda"):
                tokens  = tokens.pin_memory().to(device, non_blocking=True)
                txt_in  = txt_in.pin_memory().to(device, non_blocking=True)
                img_in  = img_in.pin_memory().to(device, non_blocking=True)
                aud_in  = aud_in.pin_memory().to(device, non_blocking=True)
                txt_prd = txt_prd.pin_memory().to(device, non_blocking=True)
                img_prd = img_prd.pin_memory().to(device, non_blocking=True)
                aud_prd = aud_prd.pin_memory().to(device, non_blocking=True)
            else:
                tokens  = tokens.to(device)
                txt_in  = txt_in.to(device)
                img_in  = img_in.to(device)
                aud_in  = aud_in.to(device)
                txt_prd = txt_prd.to(device)
                img_prd = img_prd.to(device)
                aud_prd = aud_prd.to(device)

            yield tokens, txt_in, img_in, aud_in, txt_prd, img_prd, aud_prd
