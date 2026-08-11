from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, PreTrainedTokenizerFast


def tiny_backbone(tmp_path: Path):
    tokenizer_dir = tmp_path / "tokenizer"
    vocabulary = {"<unk>": 0, "<eos>": 1, **{f"t{i}": i + 2 for i in range(98)}}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tokenizer.save_pretrained(tokenizer_dir)
    config = LlamaConfig(
        vocab_size=102,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return tokenizer_dir, config.to_dict()

