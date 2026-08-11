import math
import inspect
import json
import os
from dataclasses import asdict, dataclass, fields
from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache, Cache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


BACKBONE_ATTN_IMPLEMENTATION = os.environ.get(
    "INTERLEAVED_LM_ATTN_IMPLEMENTATION",
    "sdpa" if hasattr(F, "scaled_dot_product_attention") else "eager",
)


@dataclass
class ModelArgs:
    """
    Notation:
      B = batch size
      S = sequence length (input side: S_in = block_size)
      C = total codebooks (txt=1 + img_K + aud_K)
      K = number of codebooks for a modality
      D = backbone hidden size
      Dm = modality adapter hidden size
      Dd = codebook-transformer hidden size
    """
    # Data
    txt_vocabsize: Optional[int] = None
    img_vocabsize: Optional[int] = None
    aud_vocabsize: Optional[int] = None
    block_size: int = 2048
    txt_pad_token: Optional[int] = None
    img_pad_token: Optional[int] = None
    aud_pad_token: Optional[int] = None
    swt_token: Optional[int] = None
    predict_txt_special_tokens: bool = False
    txt_pad_loss_weight: float = 1.0
    n_img_codebooks: int = 0
    n_aud_codebooks: int = 0
    # Backbone
    backbone: str = "HuggingFaceTB/SmolLM-360M"
    backbone_revision: Optional[str] = None
    backbone_config: Optional[dict] = None
    tokenizer: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    warm_init: bool = True
    freeze_backbone: bool = False
    freeze_txt_inout: bool = False
    rope_theta: float = -1.0  # -1 keeps backbone default
    # Image adapters
    img_inadapter_n_layers: int = 6
    img_inadapter_dim: int = 576
    img_inadapter_mlp_dim: int = 1536
    img_inadapter_n_heads: int = 9
    img_inadapter_n_kvheads: int = 3
    img_outadapter_n_layers: int = 6
    img_outadapter_dim: int = 576
    img_outadapter_mlp_dim: int = 1536
    img_outadapter_n_heads: int = 9
    img_outadapter_n_kvheads: int = 3
    # Audio adapters
    aud_inadapter_n_layers: int = 6
    aud_inadapter_dim: int = 576
    aud_inadapter_mlp_dim: int = 1536
    aud_inadapter_n_heads: int = 9
    aud_inadapter_n_kvheads: int = 3
    aud_outadapter_n_layers: int = 6
    aud_outadapter_dim: int = 576
    aud_outadapter_mlp_dim: int = 1536
    aud_outadapter_n_heads: int = 9
    aud_outadapter_n_kvheads: int = 3
    # Codebook transformers
    img_codebook_transformer_layers: int = 0
    img_codebook_transformer_dim: int = 576
    img_codebook_transformer_mlp_dim: int = 1536
    img_codebook_transformer_n_heads: int = 9
    img_codebook_transformer_n_kvheads: int = 3
    aud_codebook_transformer_layers: int = 0
    aud_codebook_transformer_dim: int = 576
    aud_codebook_transformer_mlp_dim: int = 1536
    aud_codebook_transformer_n_heads: int = 9
    aud_codebook_transformer_n_kvheads: int = 3
    aud_codebook_transformer_text_prefix: bool = False
    img_codebook_weights: Optional[List[float]] = None
    aud_codebook_weights: Optional[List[float]] = None
    # Delay patterns
    img_delay_pattern: Optional[List[int]] = None
    aud_delay_pattern: Optional[List[int]] = None
    # Attention residuals
    img_attention_residual: str = "none"  # "none" | "static" | "dynamic"
    aud_attention_residual: str = "none"
    img_attention_residual_entropy_reg: float = 0.0
    aud_attention_residual_entropy_reg: float = 0.0
    img_attention_residual_norm: bool = True
    aud_attention_residual_norm: bool = True
    img_attention_residual_backbone_only: bool = False
    aud_attention_residual_backbone_only: bool = False
    tie_img_embeddings: bool = True
    tie_aud_embeddings: bool = True
    # Prediction heads
    img_head_type: str = "categorical"  # "categorical" | "bernoulli" | "flow"
    aud_head_type: str = "categorical"
    img_flow_steps: int = 20
    aud_flow_steps: int = 20
    img_flow_d: int = 64
    aud_flow_d: int = 64


def get_backbone(
    backbone: str,
    load_pretrained: bool,
    rope_theta: float,
    attn_implementation: str = BACKBONE_ATTN_IMPLEMENTATION,
    *,
    revision: Optional[str] = None,
    backbone_config: Optional[dict] = None,
    tokenizer: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    load_tokenizer: bool = True,
):
    """
    Returns:
      backbone_model: HF causal LM (embeddings still attached here)
      tokenizer: HF tokenizer
    """
    text_tokenizer = None
    if load_tokenizer:
        text_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer or backbone,
            revision=tokenizer_revision or revision,
            add_bos_token=False,
            add_eos_token=True,
        )
    if load_pretrained:
        backbone_model = AutoModelForCausalLM.from_pretrained(
            backbone,
            revision=revision,
            torch_dtype=torch.float32,
            attn_implementation=attn_implementation
        )
    else:
        if backbone_config is None:
            cfg = AutoConfig.from_pretrained(backbone, revision=revision)
        else:
            raw_config = dict(backbone_config)
            model_type = raw_config.pop("model_type")
            cfg = AutoConfig.for_model(model_type, **raw_config)
        cfg.torch_dtype = torch.float32
        cfg._attn_implementation = attn_implementation
        cfg._attn_implementation_internal = attn_implementation
        backbone_model = AutoModelForCausalLM.from_config(
            cfg, attn_implementation=attn_implementation
        )

    if rope_theta > 0 and getattr(backbone_model.config, "rope_theta", None) != rope_theta:
        print(f"Changing RoPE base frequency from {getattr(backbone_model.config, 'rope_theta', None)} to {rope_theta}")
        backbone_model.config.rope_theta = rope_theta
        if hasattr(backbone_model, "gpt_neox"):
            backbone_model.gpt_neox.rotary_emb.__init__(config=backbone_model.config)
        elif hasattr(backbone_model, "model"):
            backbone_model.model.rotary_emb.__init__(config=backbone_model.config)

    return backbone_model, text_tokenizer


class CausalLMFromTextPretrained(nn.Module):
    """
    Holds:
      - text embedding table
      - text unembedding (LM head)
      - backbone transformer (as context_model)

    forward supports small FSDP-safe ops:
      _fsdp_op="embed_text": tokens_bs -> embs_bsd
      _fsdp_op="unembed_text": h_nd -> logits_nv
    """
    def __init__(self, config: ModelArgs, backbone: str, is_resume: bool = False):
        super().__init__()
        load_pretrained = (not is_resume) and config.warm_init
        backbone_model, txt_tokenizer = get_backbone(
            backbone,
            load_pretrained,
            config.rope_theta,
            revision=config.backbone_revision,
            backbone_config=config.backbone_config,
            tokenizer=config.tokenizer,
            tokenizer_revision=config.tokenizer_revision,
            load_tokenizer=config.txt_vocabsize is not None,
        )
        backbone_cfg = backbone_model.config

        self.txt_tokenizer = txt_tokenizer
        self.dim = backbone_cfg.hidden_size
        self.n_layers = backbone_cfg.num_hidden_layers
        self.n_heads = backbone_cfg.num_attention_heads
        self.block_size = config.block_size
        self.txt_pad_token = config.txt_pad_token

        pretrained_txt_embed = backbone_model.get_input_embeddings()
        pretrained_txt_unembed = backbone_model.get_output_embeddings()

        backbone_num_embs = pretrained_txt_embed.weight.size(0)
        final_txt_vocabsize = config.txt_vocabsize

        if final_txt_vocabsize:
            if final_txt_vocabsize > backbone_num_embs:
                print(f"Expanding text vocab from {backbone_num_embs} to {final_txt_vocabsize}")
                n_new = final_txt_vocabsize - backbone_num_embs
                new_embed_weights = self.expand_pretrained_embs(pretrained_txt_embed.weight.data, n_new)

                self.txt_embed = nn.Embedding(
                    final_txt_vocabsize,
                    self.dim,
                    padding_idx=self.txt_pad_token
                )
                self.txt_embed.weight.data[:backbone_num_embs] = pretrained_txt_embed.weight.data
                self.txt_embed.weight.data[backbone_num_embs:] = new_embed_weights[backbone_num_embs:]

                self.txt_unembed = nn.Linear(self.dim, final_txt_vocabsize, bias=False)
                if backbone_cfg.tie_word_embeddings:
                    self.txt_unembed.weight = self.txt_embed.weight
                else:
                    new_unembed_weights = self.expand_pretrained_embs(pretrained_txt_unembed.weight.data, n_new)
                    self.txt_unembed.weight.data[:backbone_num_embs] = pretrained_txt_unembed.weight.data
                    self.txt_unembed.weight.data[backbone_num_embs:] = new_unembed_weights[backbone_num_embs:]
            else:
                self.txt_embed = nn.Embedding(
                    backbone_num_embs,
                    self.dim,
                    padding_idx=self.txt_pad_token
                )
                self.txt_embed.weight.data.copy_(pretrained_txt_embed.weight.data)

                self.txt_unembed = nn.Linear(self.dim, backbone_num_embs, bias=False)
                if backbone_cfg.tie_word_embeddings:
                    self.txt_unembed.weight = self.txt_embed.weight
                else:
                    self.txt_unembed.weight.data.copy_(pretrained_txt_unembed.weight.data)

        backbone_model.set_input_embeddings(None)
        backbone_model.set_output_embeddings(None)
        self.context_model = backbone_model.model
        del backbone_model

    def expand_pretrained_embs(self, emb_table_vd: torch.Tensor, num_new: int) -> torch.Tensor:
        """
        emb_table_vd: (V_old, D)
        returns updated (V_old + num_new, D)
        """
        V_old, D = emb_table_vd.shape
        avg_norm = emb_table_vd.norm(dim=1).mean()
        new_embs_nd = torch.randn(num_new, D, dtype=emb_table_vd.dtype, device=emb_table_vd.device)
        new_embs_nd *= avg_norm / new_embs_nd.norm(dim=1, keepdim=True)
        updated_vd = torch.cat([emb_table_vd, new_embs_nd], dim=0)
        print(f"Expanded embedding table from {V_old} to {updated_vd.shape[0]}")
        return updated_vd

    def forward(
        self,
        inputs_embeds_bsd: Optional[torch.Tensor] = None,  # (B,S,D)
        use_cache: bool = False,
        cache=None,
        _fsdp_op: Optional[str] = None,
        _tokens: Optional[torch.Tensor] = None
    ):
        if _fsdp_op is not None:
            if _fsdp_op == "embed_text":
                tokens_bs = _tokens
                assert tokens_bs is not None
                return self.txt_embed(tokens_bs)  # (B,S,D)
            if _fsdp_op == "unembed_text":
                h_nd = _tokens
                assert h_nd is not None
                return self.txt_unembed(h_nd)     # (N,V)
            raise ValueError(f"Unknown _fsdp_op: {_fsdp_op}")

        return self.context_model(
            inputs_embeds=inputs_embeds_bsd,
            use_cache=use_cache,
            past_key_values=cache,
        )

class RMSNormNoAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(inputs.dtype)


class AttentionResidual(nn.Module):
    """
    Static or dynamic attention residual over intermediate hidden states.

    Inputs:
      last_h_bsd:      (B,S,D) last layer hidden
      stacked_h_lbsd: (L,B,S,D) all layer hiddens (excluding embedding layer)
      preds_mask_bs:  (B,S) bool mask over time positions

    Returns:
      residual_h_bsd: (B,S,D)
      entropy:       scalar tensor (zero for static attention residuals)
    """
    def __init__(
        self,
        n_layers: int,
        dim: int,
        mode: str,
        normalize: bool = True,
    ):
        super().__init__()
        if mode not in {"static", "dynamic"}:
            raise ValueError("attention residual mode must be static or dynamic")
        self.mode = mode

        self.static_logits = nn.Parameter(0.01 * torch.randn(n_layers, 1, 1, 1))
        self.dynamic_selector = (
            nn.Linear(dim, n_layers) if mode == "dynamic" else None
        )

        self.norm = None
        if normalize:
            self.norm = RMSNormNoAffine(dim)

    def forward(
        self,
        last_h_bsd: torch.Tensor,
        stacked_h_lbsd: torch.Tensor,
        stacked_in_lbsd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert stacked_h_lbsd is not None
        if stacked_in_lbsd is not None:
            stacked_h_lbsd = torch.cat([stacked_in_lbsd, stacked_h_lbsd], dim=0)
        if self.norm is not None:
            stacked_h_lbsd = self.norm(stacked_h_lbsd)
        entropy = last_h_bsd.new_zeros(())

        weights_l111 = F.softmax(self.static_logits, dim=0)
        residual_h_bsd = (stacked_h_lbsd * weights_l111).sum(0)

        if self.dynamic_selector is not None:
            weights_bsl = F.softmax(self.dynamic_selector(residual_h_bsd), dim=-1)
            ent_bs = -(weights_bsl * (weights_bsl + 1e-12).log()).sum(-1)
            entropy = ent_bs.mean()
            weights_lbs1 = weights_bsl.permute(2, 0, 1).unsqueeze(-1)
            residual_h_bsd = (stacked_h_lbsd * weights_lbs1).sum(0)

        return residual_h_bsd, entropy


class PerceptionModule(nn.Module):
    """
    Input adapter for a modality (img/aud).

    forward:
      tokens_bsk: (B,S,K)
    returns:
      h_bsd: (B,S,D_backbone)

    If input_type != "categorical", the codebook vector is treated as a
    continuous input and projected jointly to the adapter dimension.
    """
    def __init__(
        self,
        vocabsize: int,
        pad_token: int,
        n_codebooks: int,
        backbone_dim: int,
        block_size: int,
        base_layer_type,
        base_config,
        n_in_layers: int,
        in_dim: int,
        in_mlp_dim: int,
        in_heads: int,
        in_kvheads: int,
        uses_attention_residual: bool,
        input_type: str = "categorical",  # "categorical" | "bernoulli" | "flow"
    ):
        super().__init__()
        self.vocabsize = vocabsize
        self.pad_token = pad_token
        self.n_codebooks = n_codebooks
        self.block_size = block_size
        self.in_dim = in_dim
        self.backbone_dim = backbone_dim
        self.n_in_layers = n_in_layers
        self.input_type = input_type

        if self.input_type == "categorical":
            self.embed = nn.ModuleList(
                [nn.Embedding(vocabsize, in_dim, padding_idx=pad_token) for _ in range(n_codebooks)]
            )
        else:
            self.embed = nn.Linear(n_codebooks, in_dim, bias=True)

        self.in_layers = None
        self.in_rotary = None
        if n_in_layers and n_in_layers > 0:
            cfg = deepcopy(base_config)
            adapter_attention = (
                "sdpa" if hasattr(F, "scaled_dot_product_attention") else "eager"
            )
            cfg._attn_implementation = adapter_attention
            cfg._attn_implementation_internal = adapter_attention
            assert in_dim % in_heads == 0
            cfg.head_dim = in_dim // in_heads
            cfg.hidden_size = in_dim
            cfg.intermediate_size = in_mlp_dim
            cfg.num_attention_heads = in_heads
            cfg.num_key_value_heads = in_kvheads
            cfg.max_position_embeddings = block_size

            self.in_layers = nn.ModuleList(
                [base_layer_type(cfg, i) for i in range(n_in_layers)]
            )
            self.in_rotary = LlamaRotaryEmbedding(cfg)

        self.in2backbone_proj = None
        self.in2attention_residual_proj = None
        if in_dim != backbone_dim:
            self.in2backbone_proj = nn.Linear(in_dim, backbone_dim, bias=True)
            if uses_attention_residual and self.in_layers is not None:
                self.in2attention_residual_proj = nn.Linear(
                    in_dim, backbone_dim, bias=True
                )

    @staticmethod
    def _build_chunk_causal_mask(x_mask_bs: torch.Tensor) -> torch.Tensor:
        """
        x_mask_bs : (B, S) bool   True for positions belonging to this modality input
        returns   : (B, 1, S, S) bool attention mask (True=allowed),
                    allowing attention only within same contiguous segment AND causally.
        """
        B, S = x_mask_bs.shape
        device = x_mask_bs.device
        start_bs = x_mask_bs & ~F.pad(x_mask_bs[:, :-1], (1, 0), value=False)
        seg_id_bs = torch.cumsum(start_bs, dim=1)           # (B,S)
        seg_id_bs = seg_id_bs.masked_fill(~x_mask_bs, -1)  # -1 outside modality
        same_seg_bss = seg_id_bs.unsqueeze(1) == seg_id_bs.unsqueeze(2)  # (B,S,S)
        causal_1ss = torch.tril(
            torch.ones(1, S, S, dtype=torch.bool, device=device)
        )  # (1,S,S)
        allow_bss = same_seg_bss & causal_1ss  # (B,S,S)
        return allow_bss.unsqueeze(1)          # (B,1,S,S)

    def forward(
        self,
        tokens_bsk: torch.Tensor,                  # (B,S,K)
        input_masks_bs: torch.Tensor,  # (B,S) bool, True where modality is present
        return_hidden: bool = False,
        use_cache: bool = False,
        cache: Optional[Cache] = None,
    ):
        B, S, K = tokens_bsk.shape
        assert K == self.n_codebooks, f"Input number of codebooks {K} doesn't match PerceptionModule's expected {self.n_codebooks}"
        next_cache = None

        h_bsd = 0
        if self.input_type == "categorical":
            for k in range(self.n_codebooks):
                h_bsd = h_bsd + self.embed[k](tokens_bsk[:, :, k])  # (B,S,in_dim)
        else:
            x_bsk = tokens_bsk.float()
            h_bsd = self.embed(x_bsk)  # (B,S,in_dim)

        stacked_in = None
        if self.in_layers is not None:
            cache_obj: Optional[Cache] = cache
            if use_cache:
                if cache_obj is None:
                    cache_obj = DynamicCache()
                past_len = int(cache_obj.get_seq_length())
            else:
                past_len = 0

            device = tokens_bsk.device
            pos_bs = torch.arange(past_len, past_len + S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)  # (1,S)
            attn_mask_b1ss = (
                self._build_chunk_causal_mask(input_masks_bs)
                if not use_cache or past_len == 0
                else None
            )
            position_embeddings = self.in_rotary(h_bsd, pos_bs)
            hs = []
            for layer in self.in_layers:
                layer_out = layer(
                    h_bsd,
                    attention_mask=attn_mask_b1ss,
                    position_ids=pos_bs,
                    use_cache=use_cache,
                    past_key_value=cache_obj,
                    position_embeddings=position_embeddings,
                )
                h_bsd = layer_out[0]
                if return_hidden:
                    hs.append(h_bsd)
            if return_hidden:
                stacked_in = torch.stack(hs, dim=0)  # (L_in,B,S,in_dim)
            next_cache = cache_obj if use_cache else None

        if self.in2backbone_proj is not None:
            h_bsd = self.in2backbone_proj(h_bsd)
            if stacked_in is not None and self.in2attention_residual_proj is not None:
                stacked_in = torch.stack(
                    [self.in2attention_residual_proj(h) for h in stacked_in], dim=0
                )  # (L_in,B,S,D)

        if use_cache:
            return h_bsd, stacked_in, next_cache
        if return_hidden:
            return h_bsd, stacked_in
        return h_bsd


class ExpressionModule(nn.Module):
    """
    Output head for a modality (img/aud).

    forward inputs:
      last_h_bsd:      (B,S,D_backbone)
      preds_mask_bs:   (B,S) boolean mask over time positions
      target_tokens_bsk:(B,S,K) same-frame targets for depth teacher forcing
      stacked_h_lbsd:  (L,B,S,D_backbone) for attention residuals

    returns:
      logits_nkv: (N,K,V)
      entropy:    scalar (zero unless the attention residual is dynamic)
    """
    def __init__(
        self,
        vocabsize: int,
        pad_token: int,
        n_codebooks: int,
        backbone_dim: int,
        block_size: int,
        base_layer_type,
        base_config,
        perception_embed,
        n_in_layers: int,
        n_out_layers: int,
        out_dim: int,
        out_mlp_dim: int,
        out_heads: int,
        out_kvheads: int,
        codebookt_layers: int,
        codebookt_dim: int,
        codebookt_mlp_dim: int,
        codebookt_heads: int,
        codebookt_kvheads: int,
        codebookt_text_prefix: bool = False,
        tie_embeddings: bool = True,
        attention_residual: str = "none",
        attention_residual_norm: bool = True,
        attention_residual_entropy_reg: float = 0.0,
        head_type: str = "categorical",
        flow_steps: int = 20,
        flow_dim: int = 64,
        attention_residual_backbone_only: bool = False,
    ):
        super().__init__()
        self.vocabsize = vocabsize
        self.pad_token = pad_token
        self.n_codebooks = n_codebooks
        self.block_size = block_size
        self.attention_residual_entropy_reg = attention_residual_entropy_reg
        self.perception_embed = perception_embed
        if isinstance(perception_embed, nn.ModuleList):
            self.in_dim = perception_embed[0].embedding_dim            
        else:
            self.in_dim = perception_embed.out_features
        self.head_type = head_type
        self.flow_steps = flow_steps

        if head_type != "categorical":
            codebookt_layers = 0

        self.attention_residual_backbone_only = attention_residual_backbone_only
        self.attention_residual = None
        if attention_residual != "none":
            L_backbone = base_config.num_hidden_layers
            L_total = (
                L_backbone
                if attention_residual_backbone_only
                else L_backbone + (n_in_layers or 0)
            )
            self.attention_residual = AttentionResidual(
                n_layers=L_total,
                dim=backbone_dim,
                mode=attention_residual,
                normalize=attention_residual_norm,
            )

        self.out_layers = None
        self.out_rotary = None
        if n_out_layers and n_out_layers > 0:
            cfg = deepcopy(base_config)
            assert out_dim % out_heads == 0
            cfg.head_dim = out_dim // out_heads
            cfg.hidden_size = out_dim
            cfg.intermediate_size = out_mlp_dim
            cfg.num_attention_heads = out_heads
            cfg.num_key_value_heads = out_kvheads
            cfg.max_position_embeddings = block_size

            self.out_layers = nn.ModuleList(
                [base_layer_type(cfg, i) for i in range(n_out_layers)]
            )
            self.out_rotary = LlamaRotaryEmbedding(cfg)

        self.backbone2out_proj = None
        if backbone_dim != out_dim:
            self.backbone2out_proj = nn.Linear(backbone_dim, out_dim, bias=False)

        self.codebookt_layers = None
        self.codebookt_rotary = None
        self.out2in_proj = None
        self.in2codebookt_proj = None
        self.codebookt_text_prefix = codebookt_text_prefix
        self.text2in_proj = None

        codebookt_out_dim = out_dim  # default output dim if no codebook transformer

        if codebookt_layers and codebookt_layers > 0 and n_codebooks > 1:
            cfg = deepcopy(base_config)
            assert codebookt_dim % codebookt_heads == 0
            cfg.head_dim = codebookt_dim // codebookt_heads
            cfg.hidden_size = codebookt_dim
            cfg.intermediate_size = codebookt_mlp_dim
            cfg.num_attention_heads = codebookt_heads
            cfg.num_key_value_heads = codebookt_kvheads
            cfg.max_position_embeddings = n_codebooks

            self.codebookt_layers = nn.ModuleList(
                [base_layer_type(cfg, i) for i in range(codebookt_layers)]
            )
            self.codebookt_rotary = LlamaRotaryEmbedding(cfg)

            if out_dim != self.in_dim:
                self.out2in_proj = nn.Linear(out_dim, self.in_dim, bias=False)
            if self.in_dim != codebookt_dim:
                self.in2codebookt_proj = nn.Linear(self.in_dim, codebookt_dim, bias=False)
            if codebookt_text_prefix and backbone_dim != self.in_dim:
                self.text2in_proj = nn.Linear(backbone_dim, self.in_dim, bias=False)

            codebookt_out_dim = codebookt_dim

        if self.head_type == "categorical":
            self.unembed = nn.ModuleList(
                [nn.Linear(codebookt_out_dim, vocabsize, bias=False) for _ in range(n_codebooks)]
            )
            if tie_embeddings:
                for k in range(n_codebooks):
                    if self.perception_embed[k].weight.shape != self.unembed[k].weight.shape:
                        raise ValueError(
                            "tied modality embeddings require matching input and output dimensions"
                        )
                    self.unembed[k].weight = self.perception_embed[k].weight

        elif self.head_type == "bernoulli":
            self.unembed = nn.Linear(codebookt_out_dim, n_codebooks, bias=True)
            
        elif self.head_type == "flow":
            if flow_steps < 1:
                raise ValueError("flow_steps must be at least one")
            d_in = codebookt_out_dim
            self.flow_ctx_proj = nn.Linear(d_in, flow_dim)   # (D_in -> D_flow)
            self.flow_xt_proj  = nn.Linear(self.n_codebooks + 1, flow_dim)
            self.flow_hidden   = nn.Linear(flow_dim, flow_dim)
            self.flow_act      = nn.SiLU()
            self.flow_out      = nn.Linear(flow_dim, self.n_codebooks)
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

    def forward_with_attention_residual(
        self,
        h_bsd: torch.Tensor,
        stacked_h_lbsd: Optional[torch.Tensor] = None,
        stacked_in_lbsd: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache: Optional[Cache] = None
    ):
        entropy = h_bsd.new_zeros(())
        next_cache = [] if use_cache else None
        if self.attention_residual is not None:
            if self.attention_residual_backbone_only:
                stacked_in_lbsd = None
            h_bsd, entropy = self.attention_residual(
                h_bsd, stacked_h_lbsd, stacked_in_lbsd
            )

        if self.backbone2out_proj is not None:
            h_bsd = self.backbone2out_proj(h_bsd)

        next_cache = None
        if self.out_layers is not None:
            B, S, _ = h_bsd.size()
            cache_obj: Optional[Cache] = cache
            if use_cache:
                if cache_obj is None:
                    cache_obj = DynamicCache()
                past_len = int(cache_obj.get_seq_length())
            else:
                past_len = 0
            device = h_bsd.device
            pos_bs = torch.arange(past_len, past_len + S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)  # (B,S)
            position_embeddings = self.out_rotary(h_bsd, pos_bs)
            for layer in self.out_layers:
                layer_out = layer(
                    h_bsd,
                    position_ids=pos_bs,
                    use_cache=use_cache,
                    past_key_value=cache_obj,
                    position_embeddings=position_embeddings,
                )
                h_bsd = layer_out[0]

            next_cache = cache_obj if use_cache else None
        if use_cache:
            return h_bsd, next_cache, entropy
        return h_bsd, entropy

    def flow_velocity(
        self,
        h_ctx_nd: torch.Tensor,  # (N,D)
        x_nk: torch.Tensor,      # (N,K)
        t_n: torch.Tensor        # (N,)
    ) -> torch.Tensor:
        """
        Joint flow velocity v(x,t|h)
        h_ctx_nd : (N,D)   context features per position
        x_nk     : (N,K)   current state (one K-dim vector per position)
        t_n      : (N,)    time values for each position
        returns:
          v_nk   : (N,K)
        """
        N, K = x_nk.shape
        assert K == self.n_codebooks, f"Expected {self.n_codebooks} dims, got {K}"
        ctx_feat_nd = self.flow_ctx_proj(h_ctx_nd)  # (N,D_flow)
        t_n1 = t_n.view(N, 1)                        # (N,1)
        xt_in_n1 = torch.cat([x_nk, t_n1], dim=1)    # (N,K+1)
        xt_feat_nd = self.flow_xt_proj(xt_in_n1)     # (N,D_flow)
        hidden_nd = self.flow_act(ctx_feat_nd + xt_feat_nd)      # (N,D_flow)
        hidden_nd = self.flow_act(self.flow_hidden(hidden_nd))   # (N,D_flow)
        v_nk = self.flow_out(hidden_nd)  # (N,K)
        return v_nk

    def categorical_logits(
        self,
        h_pred_nd: torch.Tensor,
        target_tokens_nk: Optional[torch.Tensor],
        text_prefix_nd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project categorical states exactly as in teacher-forced training."""
        if self.codebookt_layers is None:
            return torch.stack(
                [self.unembed[k](h_pred_nd) for k in range(self.n_codebooks)],
                dim=1,
            )
        if target_tokens_nk is None:
            raise ValueError("depth-transformer heads require same-frame target codes")

        previous_embs_nkd = torch.stack(
            [
                self.perception_embed[k](target_tokens_nk[:, k])
                for k in range(self.n_codebooks)
            ],
            dim=1,
        )
        h_in_nd = self.out2in_proj(h_pred_nd) if self.out2in_proj is not None else h_pred_nd
        if self.codebookt_text_prefix:
            if text_prefix_nd is None:
                raise ValueError("text-prefixed codebook decoding requires aligned text embeddings")
            prefix_nd = (
                self.text2in_proj(text_prefix_nd)
                if self.text2in_proj is not None
                else text_prefix_nd
            )
            first_n1d = prefix_nd[:, None]
        else:
            first_n1d = torch.zeros_like(h_in_nd[:, None])
        shifted_previous_nkd = torch.cat(
            [first_n1d, previous_embs_nkd[:, :-1]], dim=1
        )
        codebookt_in_nkd = h_in_nd[:, None] + shifted_previous_nkd
        if self.in2codebookt_proj is not None:
            codebookt_in_nkd = self.in2codebookt_proj(codebookt_in_nkd)

        codebook_ids_1k = torch.arange(
            self.n_codebooks, device=codebookt_in_nkd.device
        ).unsqueeze(0)
        h_codebookt_nkd = codebookt_in_nkd
        position_embeddings = self.codebookt_rotary(
            h_codebookt_nkd, codebook_ids_1k
        )
        for layer in self.codebookt_layers:
            h_codebookt_nkd = layer(
                h_codebookt_nkd,
                position_ids=codebook_ids_1k,
                position_embeddings=position_embeddings,
            )[0]
        return torch.stack(
            [
                self.unembed[k](h_codebookt_nkd[:, k])
                for k in range(self.n_codebooks)
            ],
            dim=1,
        )

    
    def forward(
        self,
        last_h_bsd: torch.Tensor,
        preds_mask_bs: torch.Tensor,
        target_tokens_bsk: Optional[torch.Tensor],
        stacked_h_lbsd: Optional[torch.Tensor] = None,
        stacked_in_lbsd: Optional[torch.Tensor] = None,
        flow_xt_nk: Optional[torch.Tensor] = None,
        flow_t_n: Optional[torch.Tensor] = None,
        text_prefix_bsd: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h_bsd, entropy = self.forward_with_attention_residual(
            last_h_bsd, stacked_h_lbsd, stacked_in_lbsd
        )
        h_pred_nd = h_bsd[preds_mask_bs]  # (N, Dout)

        K = self.n_codebooks

        if self.head_type == "categorical":
            if self.codebookt_layers is not None:
                B, S, K = target_tokens_bsk.shape
                flat_tokens_ak = target_tokens_bsk.reshape(B * S, K)
                target_tokens_nk = flat_tokens_ak[preds_mask_bs.reshape(-1)]
            else:
                target_tokens_nk = None
            text_prefix_nd = (
                text_prefix_bsd[preds_mask_bs]
                if text_prefix_bsd is not None
                else None
            )
            logits_nkv = self.categorical_logits(
                h_pred_nd, target_tokens_nk, text_prefix_nd
            )
            vocab_ids = torch.arange(self.vocabsize, device=logits_nkv.device)
            logits_nkv = logits_nkv.masked_fill(
                vocab_ids == self.pad_token, -float("inf")
            )
            return logits_nkv, entropy
        
        if self.head_type == "bernoulli":
            logits_nk1 = self.unembed(h_pred_nd).unsqueeze(-1)  # (N,K,1)
            return logits_nk1, entropy

        assert flow_xt_nk is not None and flow_t_n is not None
        v_nk = self.flow_velocity(h_pred_nd, flow_xt_nk, flow_t_n)  # (N,K)
        return v_nk, entropy


@dataclass
class _ModSpec:
    name: str
    enabled: bool
    perception: Optional[nn.Module]
    expression: Optional[nn.Module]
    sl: slice
    pad: int
    vocab: int
    K: int
    delay: List[int]


class PerceptionExpressionAdaptedTextLM(nn.Module):
    """
    Main multimodal LM.

    forward inputs (dataset-aligned):
      tokens_bsc:           (B,S,C)  S=block_size+1
      txt_input_masks_bs:   (B,S)
      img_input_masks_bs:   (B,S)
      aud_input_masks_bs:   (B,S)
      *_preds_masks_bs_in:  (B,S-1) or None   (already shifted in collate_fn)

    returns:
      logits_txt_nv or None
      logits_img_nkv or None
      logits_aud_nkv or None
    """
    last_loss: Optional[torch.Tensor]
    losses: Optional[dict]

    def _build_modality(
        self,
        config: ModelArgs,
        pref: str,                         # "img" or "aud"
        sl: slice,
        delay_pattern: List[int],
        base_layer_type,
        base_config,
    ):
        """
        Factory for PerceptionModule + ExpressionModule with prefix-based config.
        Returns a _ModSpec. Does NOT touch any other state.
        """
        vocab = getattr(config, f"{pref}_vocabsize", None)
        K = getattr(config, f"n_{pref}_codebooks", 0) or 0
        enabled = (vocab is not None) and (K > 0)
        pad = getattr(config, f"{pref}_pad_token")
        head_type = getattr(config, f"{pref}_head_type")

        if not enabled:
            return _ModSpec(
                name=pref, enabled=False,
                perception=None, expression=None,
                sl=sl, pad=pad, vocab=vocab, K=K,
                delay=delay_pattern,
            )

        perception = PerceptionModule(
            vocabsize=vocab,
            pad_token=pad,
            n_codebooks=K,
            backbone_dim=self.dim,
            block_size=self.block_size,
            base_layer_type=base_layer_type,
            base_config=base_config,
            n_in_layers=getattr(config, f"{pref}_inadapter_n_layers"),
            in_dim=getattr(config, f"{pref}_inadapter_dim"),
            in_mlp_dim=getattr(config, f"{pref}_inadapter_mlp_dim"),
            in_heads=getattr(config, f"{pref}_inadapter_n_heads"),
            in_kvheads=getattr(config, f"{pref}_inadapter_n_kvheads"),
            uses_attention_residual=(
                getattr(config, f"{pref}_attention_residual") != "none"
                and not getattr(
                    config, f"{pref}_attention_residual_backbone_only"
                )
            ),
            input_type=head_type,
        )

        expression = ExpressionModule(
            vocabsize=vocab,
            pad_token=pad,
            n_codebooks=K,
            backbone_dim=self.dim,
            block_size=self.block_size,
            base_layer_type=base_layer_type,
            base_config=base_config,
            perception_embed=perception.embed,
            n_in_layers=getattr(config, f"{pref}_inadapter_n_layers"),
            n_out_layers=getattr(config, f"{pref}_outadapter_n_layers"),
            out_dim=getattr(config, f"{pref}_outadapter_dim"),
            out_mlp_dim=getattr(config, f"{pref}_outadapter_mlp_dim"),
            out_heads=getattr(config, f"{pref}_outadapter_n_heads"),
            out_kvheads=getattr(config, f"{pref}_outadapter_n_kvheads"),
            codebookt_layers=getattr(config, f"{pref}_codebook_transformer_layers"),
            codebookt_dim=getattr(config, f"{pref}_codebook_transformer_dim"),
            codebookt_mlp_dim=getattr(config, f"{pref}_codebook_transformer_mlp_dim"),
            codebookt_heads=getattr(config, f"{pref}_codebook_transformer_n_heads"),
            codebookt_kvheads=getattr(config, f"{pref}_codebook_transformer_n_kvheads"),
            codebookt_text_prefix=(
                pref == "aud" and config.aud_codebook_transformer_text_prefix
            ),
            tie_embeddings=getattr(config, f"tie_{pref}_embeddings"),
            attention_residual=getattr(config, f"{pref}_attention_residual"),
            attention_residual_norm=getattr(
                config, f"{pref}_attention_residual_norm"
            ),
            attention_residual_entropy_reg=getattr(
                config, f"{pref}_attention_residual_entropy_reg"
            ),
            head_type=head_type,
            flow_steps=getattr(config, f"{pref}_flow_steps"),
            flow_dim=getattr(config, f"{pref}_flow_d"),
            attention_residual_backbone_only=getattr(
                config, f"{pref}_attention_residual_backbone_only"
            ),
        )

        return _ModSpec(
            name=pref, enabled=True,
            perception=perception, expression=expression,
            sl=sl, pad=pad, vocab=vocab, K=K,
            delay=delay_pattern,
        )


    def __init__(self, config: ModelArgs, is_resume: bool = False):
        super().__init__()
        for modality in ("img", "aud"):
            mode = getattr(config, f"{modality}_attention_residual")
            if mode not in {"none", "static", "dynamic"}:
                raise ValueError(
                    f"{modality}_attention_residual must be none, static, or dynamic"
                )
        self.config = config

        self.models_txt = config.txt_vocabsize is not None
        self.n_img_codebooks = config.n_img_codebooks or 0
        self.n_aud_codebooks = config.n_aud_codebooks or 0
        self.models_img = config.img_vocabsize is not None and self.n_img_codebooks > 0
        self.models_aud = config.aud_vocabsize is not None and self.n_aud_codebooks > 0

        for modality, count in (
            ("img", self.n_img_codebooks),
            ("aud", self.n_aud_codebooks),
        ):
            weights = getattr(config, f"{modality}_codebook_weights")
            if weights is not None and (
                len(weights) != count or any(weight <= 0 for weight in weights)
            ):
                raise ValueError(
                    f"{modality}_codebook_weights must contain one positive value per codebook"
                )

        if config.txt_pad_loss_weight <= 0:
            raise ValueError("txt_pad_loss_weight must be positive")
        if config.aud_codebook_transformer_text_prefix and not (
            self.models_txt
            and self.models_aud
            and config.aud_head_type == "categorical"
            and config.aud_codebook_transformer_layers > 0
        ):
            raise ValueError(
                "text-prefixed audio decoding requires text, categorical audio, and a codebook transformer"
            )

        self.global_workspace = CausalLMFromTextPretrained(config, config.backbone, is_resume)
        base_layer_type = type(self.global_workspace.context_model.layers[0])
        base_config = self.global_workspace.context_model.config

        self.dim = self.global_workspace.dim
        self.n_layers = self.global_workspace.n_layers
        self.n_heads = self.global_workspace.n_heads
        self.block_size = config.block_size

        self.txt_pad_token = config.txt_pad_token
        self.swt_token = config.swt_token
        self.img_pad_token = config.img_pad_token
        self.aud_pad_token = config.aud_pad_token

        if self.models_txt and ((self.models_img or self.models_aud)):
            assert self.txt_pad_token is not None, "txt_pad_token must be set when txt_vocabsize is not None."
        if self.models_img:
            assert self.img_pad_token is not None, "img_pad_token must be set when img_vocabsize is not None."
        if self.models_aud:
            assert self.aud_pad_token is not None, "aud_pad_token must be set when aud_vocabsize is not None."

        self.n_codebooks = (1 if self.models_txt else 0) + self.n_img_codebooks + self.n_aud_codebooks
        start = 1 if self.models_txt else 0
        self.img_slice = slice(start, start + self.n_img_codebooks)
        self.aud_slice = slice(start + self.n_img_codebooks, start + self.n_img_codebooks + self.n_aud_codebooks)

        self.img_delay_pattern = (
            config.img_delay_pattern
            if config.img_delay_pattern is not None
            else list(range(self.n_img_codebooks))
        )
        self.aud_delay_pattern = (
            config.aud_delay_pattern
            if config.aud_delay_pattern is not None
            else list(range(self.n_aud_codebooks))
        )
        for name, count, pattern in (
            ("image", self.n_img_codebooks, self.img_delay_pattern),
            ("audio", self.n_aud_codebooks, self.aud_delay_pattern),
        ):
            if len(pattern) != count or any(not isinstance(delay, int) or delay < 0 for delay in pattern):
                raise ValueError(
                    f"{name} delay pattern must contain one non-negative integer per codebook"
                )

        self._mods = {}
        for pref, sl, delay in [
            ("img", self.img_slice, self.img_delay_pattern),
            ("aud", self.aud_slice, self.aud_delay_pattern),
        ]:
            m = self._build_modality(config, pref, sl, delay, base_layer_type, base_config)
            self._mods[pref] = m
            setattr(self, f"{pref}_perception", m.perception)
            setattr(self, f"{pref}_expression", m.expression)

        self.img_perception = self._mods["img"].perception
        self.img_expression = self._mods["img"].expression
        self.aud_perception = self._mods["aud"].perception
        self.aud_expression = self._mods["aud"].expression

        if config.freeze_backbone:
            for p in self.global_workspace.context_model.parameters():
                p.requires_grad = False
        if self.models_txt and config.freeze_txt_inout:
            for p in self.global_workspace.txt_embed.parameters():
                p.requires_grad = False
            for p in self.global_workspace.txt_unembed.parameters():
                p.requires_grad = False

        self.last_loss = None
        self.losses = {
            "lm": 0.0,
            "txt": 0.0,
            "img": 0.0,
            "aud": 0.0,
            "img_attention_residual_entropy": 0.0,
            "aud_attention_residual_entropy": 0.0,
        }

        txt_params, img_params, aud_params = [], [], []
        for name, p in self.named_parameters():
            lname = name.lower()
            if "img_" in lname:
                img_params.append(p)
            elif "aud_" in lname:
                aud_params.append(p)
            else:
                txt_params.append(p)

        print(f"num of text params:  {sum(p.numel() for p in txt_params):,}")
        print(f"num of image params: {sum(p.numel() for p in img_params):,}")
        print(f"num of audio params: {sum(p.numel() for p in aud_params):,}")


    def apply_delay_pattern(self, tokens_bsk: torch.Tensor, pad_token: int, delay_pattern: List[int]):
        """
        tokens_bsk: (B,S,K)
        returns: (B,S,K) delayed along S per codebook
        """
        if tokens_bsk is None:
            return None
        B, S, K = tokens_bsk.shape
        assert len(delay_pattern) == K
        if K <= 1:
            return tokens_bsk
        positions = torch.arange(S, device=tokens_bsk.device).view(1, S, 1)
        delays = torch.tensor(delay_pattern, device=tokens_bsk.device).view(1, 1, K)
        source = (positions - delays).clamp_min(0).expand(B, S, K)
        shifted = tokens_bsk.gather(1, source)
        return torch.where(positions >= delays, shifted, pad_token)

    def dummy_ops(self, txt_preds_masks_bs: Optional[torch.Tensor],
                  img_preds_masks_bs: Optional[torch.Tensor],
                  aud_preds_masks_bs: Optional[torch.Tensor]):
        """
        Ensures parameter participation in FSDP even when a modality has no predictions.
        """
        dummy = 0.0
        if self.models_txt and txt_preds_masks_bs is None:
            for p in self.global_workspace.txt_embed.parameters():
                dummy += p.sum() * 0.0
            for p in self.global_workspace.txt_unembed.parameters():
                dummy += p.sum() * 0.0
        if self.models_img and img_preds_masks_bs is None:
            for p in self.img_perception.parameters():
                dummy += p.sum() * 0.0
            for p in self.img_expression.parameters():
                dummy += p.sum() * 0.0
        if self.models_aud and aud_preds_masks_bs is None:
            for p in self.aud_perception.parameters():
                dummy += p.sum() * 0.0
            for p in self.aud_expression.parameters():
                dummy += p.sum() * 0.0
        return dummy

    def _prep_modality_io(
        self,
        tokens_bsc: torch.Tensor,
        m: _ModSpec,
        normalize_flow_in: bool = True,
        sample_bernoulli_inputs: bool = False,
    ):
        """
        Shared img/aud stream prep.
        - delay FULL stream if categorical + K>1
        - split into teacher-forced inputs / targets
        - bernoulli:
            * targets are always binarized deterministically
            * inputs are either binarized (eval / NLL / gen) or
              sampled from normalized values in [0,1] (training)
        - normalize teacher-forced inputs for flow
        """
        full_bsk = tokens_bsc[:, :, m.sl]  # (B,S,K)

        head = m.expression.head_type
        if m.K > 1 and head == "categorical":
            full_bsk = self.apply_delay_pattern(full_bsk, m.pad, m.delay)

        in_bsk  = full_bsk[:, :-1, :]  # (B,S_in,K)
        tgt_bsk = full_bsk[:,  1:, :]  # (B,S_in,K)

        if head == "bernoulli":
            thr = 0.5 if m.vocab <= 1 else 0.5 * float(m.vocab - 1)
            if sample_bernoulli_inputs:
                probs_bsk = self._to_unit(in_bsk, m.vocab).clamp(0.0, 1.0)
                in_bsk = (torch.rand_like(probs_bsk) < probs_bsk).long()
            else:
                in_bsk = (in_bsk.float() > thr).long()
            tgt_bsk = (tgt_bsk.float() > thr).long()

        elif head == "flow" and normalize_flow_in:
            in_bsk = self._to_unit(in_bsk, m.vocab)

        return in_bsk, tgt_bsk

    def _embed_modality(
        self,
        m: _ModSpec,
        in_bsk: torch.Tensor,
        in_mask_bs: torch.Tensor,
        use_cache: bool = False,
        cache: Optional[List] = None
    ):
        """
        Shared embedding path.
        Returns:
        h_mod_bsd: (B,S_in,D)
        in_hidden: Optional, passed to the expression attention residual
        """
        want_in_hidden = m.expression.attention_residual is not None
        out = m.perception(
            in_bsk,
            input_masks_bs=in_mask_bs,
            return_hidden=want_in_hidden,
            use_cache=use_cache,
            cache=cache
        )
        next_cache = None
        if use_cache:
            if want_in_hidden:
                h_mod_bsd, in_hidden, next_cache = out
            else:
                h_mod_bsd, _, next_cache = out
                in_hidden = None
        else:
            if want_in_hidden:
                h_mod_bsd, in_hidden = out
            else:
                h_mod_bsd, in_hidden = out, None

        h_mod_bsd = h_mod_bsd * in_mask_bs.unsqueeze(-1).to(h_mod_bsd.dtype)
        if use_cache:
            return h_mod_bsd, in_hidden, next_cache
        return h_mod_bsd, in_hidden

    def _modality_logits_and_loss(
        self,
        m: _ModSpec,
        h_ctx_bsd: torch.Tensor,
        preds_mask_bs: Optional[torch.Tensor],
        in_bsk: Optional[torch.Tensor],
        tgt_bsk: Optional[torch.Tensor],
        stacked_h_lbsd: Optional[torch.Tensor],
        stacked_in_lbsd: Optional[torch.Tensor],
        codebook_weights: Optional[List[float]],
        compute_loss: bool,
        text_prefix_bsd: Optional[torch.Tensor] = None,
    ):
        """
        Shared img/aud loss+logits.

        Semantics:
        - Work on positions N where preds_mask_bs == True.
        - For each position n:
            * categorical: codebook-weighted average across active codebooks.
            * bernoulli / flow: no padding semantics; all K channels at those
              positions are valid. "Where modality is not present" should be
              encoded in preds_mask_bs, not pad_token.
        """
        if preds_mask_bs is None:
            z = h_ctx_bsd.new_zeros(())
            return None, z, z

        head = m.expression.head_type
        loss = h_ctx_bsd.new_zeros(())
        entropy = h_ctx_bsd.new_zeros(())

        if head == "flow":
            tgt_nk = tgt_bsk[preds_mask_bs]                      # (N,K)
            x0_nk = self._to_unit(tgt_nk, m.vocab)               # (N,K)

            N, K = x0_nk.shape
            t_n = torch.rand((N,), device=x0_nk.device, dtype=x0_nk.dtype)
            eps_nk = torch.randn_like(x0_nk)
            xt_nk = (1 - t_n[:, None]) * x0_nk + t_n[:, None] * eps_nk
            v_tgt_nk = eps_nk - x0_nk                             # (N,K)

            v_pred_nk, entropy = m.expression(
                h_ctx_bsd, preds_mask_bs, None,
                stacked_h_lbsd=stacked_h_lbsd,
                stacked_in_lbsd=stacked_in_lbsd,
                flow_xt_nk=xt_nk, flow_t_n=t_n
            )  # (N,K)

            mse_nk = (v_pred_nk - v_tgt_nk) ** 2                  # (N,K)
            loss = mse_nk.sum() / max(mse_nk.numel(), 1)

            if not self.training:
                self.losses[m.name] = loss.detach()
                self.losses[
                    f"{m.name}_attention_residual_entropy"
                ] = entropy.detach()
            return v_pred_nk, loss, entropy

        preds_nkv, entropy = m.expression(
            h_ctx_bsd, preds_mask_bs, tgt_bsk,
            stacked_h_lbsd=stacked_h_lbsd,
            stacked_in_lbsd=stacked_in_lbsd,
            text_prefix_bsd=text_prefix_bsd,
        )

        if not compute_loss:
            return preds_nkv, loss, entropy

        tgt_nk = tgt_bsk[preds_mask_bs]  # (N,K)

        if head == "categorical":
            mask_nk = (tgt_nk != m.pad)  # (N,K) bool

            w_k = codebook_weights or [1.0] * m.K
            assert len(w_k) == m.K
            w_k_t = h_ctx_bsd.new_tensor(w_k)  # (K,)

            per_pos = []  # list of (N,) per-codebook per-position losses
            for k in range(m.K):
                ce_n = F.cross_entropy(
                    preds_nkv[:, k], tgt_nk[:, k],
                    reduction="none", ignore_index=m.pad
                )  # (N,)
                per_pos.append(ce_n)

                denom_k = mask_nk[:, k].sum().clamp_min(1)
                lk = ce_n[mask_nk[:, k]].sum() / denom_k

                if not self.training:
                    self.losses[f"{m.name}_k{k}"] = lk.detach()

            per_pos_nk = torch.stack(per_pos, 1)  # (N,K)

            w_nk = w_k_t.view(1, m.K).expand_as(per_pos_nk) * mask_nk.to(per_pos_nk.dtype)  # (N,K)
            w_sum_n = w_nk.sum(-1, keepdim=True).clamp_min(1e-8)                            # (N,1)
            alpha_nk = w_nk / w_sum_n                                                       # (N,K)
            loss_pos_n = (per_pos_nk * alpha_nk).sum(-1)                                    # (N,)

            valid_n = mask_nk.any(-1)                                                       # (N,)
            denom = valid_n.sum().clamp_min(1)
            loss = loss_pos_n[valid_n].sum() / denom

            if not self.training:
                self.losses[m.name] = loss.detach()
                self.losses[
                    f"{m.name}_attention_residual_entropy"
                ] = entropy.detach()
            return preds_nkv, loss, entropy

        assert head == "bernoulli"
        logits_nk = preds_nkv[..., 0]   # (N,K)
        y_nk = tgt_nk.float()           # (N,K)
        per_value_loss = F.binary_cross_entropy_with_logits(
            logits_nk.view(-1),
            y_nk.view(-1),
            reduction="none",
        )
        loss = per_value_loss.sum() / max(per_value_loss.numel(), 1)

        if not self.training:
            self.losses[m.name] = loss.detach()
            self.losses[f"{m.name}_attention_residual_entropy"] = entropy.detach()
        return preds_nkv, loss, entropy

    def _modality_nll(
        self,
        m: _ModSpec,
        h_ctx_bsd: torch.Tensor,
        stacked_h_lbsd: Optional[torch.Tensor],
        stacked_in_lbsd: Optional[torch.Tensor],
        preds_mask_bs: Optional[torch.Tensor],
        in_bsk: Optional[torch.Tensor],
        tgt_bsk: Optional[torch.Tensor],
        codebook_weights: Optional[List[float]],
        text_prefix_bsd: Optional[torch.Tensor] = None,
    ):
        """
        Shared img/aud NLL per sample (aligned with forward).
        For bernoulli / flow heads we do NOT use pad semantics. Positions
        where the modality should not contribute must be masked out by
        preds_mask_bs upstream.
        """
        B = h_ctx_bsd.size(0)
        if preds_mask_bs is None or not preds_mask_bs.any():
            return h_ctx_bsd.new_zeros((B,), dtype=torch.float)

        head = m.expression.head_type
        if head == "flow":
            return self._flow_nll_per_sample(
                m.expression, h_ctx_bsd, stacked_h_lbsd, stacked_in_lbsd,
                preds_mask_bs, tgt_bsk, vocabsize=m.vocab,
            )

        with torch.no_grad():
            logits_nkv, _ = m.expression(
                h_ctx_bsd, preds_mask_bs, tgt_bsk,
                stacked_h_lbsd=stacked_h_lbsd,
                stacked_in_lbsd=stacked_in_lbsd,
                text_prefix_bsd=text_prefix_bsd,
            )
            tgt_nk = tgt_bsk[preds_mask_bs]  # (N,K)

            if head == "categorical":
                w_k = codebook_weights or [1.0] * m.K
                assert len(w_k) == m.K
                w_k_t = h_ctx_bsd.new_tensor(w_k)  # (K,)

                per_pos = []
                for k in range(m.K):
                    nll_n = F.cross_entropy(
                        logits_nkv[:, k], tgt_nk[:, k],
                        reduction="none", ignore_index=m.pad
                    )  # (N,)
                    per_pos.append(nll_n)

                per_pos_nk = torch.stack(per_pos, 1)  # (N,K)

            elif head == "bernoulli":
                logits_nk = logits_nkv[..., 0]                       # (N,K)
                y_nk = tgt_nk.float()                                # (N,K)
                per_pos_nk = F.binary_cross_entropy_with_logits(
                    logits_nk, y_nk, reduction="none"
                )  # (N,K)
            else:
                raise ValueError(f"Unknown head_type: {head}")

        lens_b = preds_mask_bs.sum(1)
        splits_loss = torch.split(per_pos_nk, tuple(lens_b.tolist()))
        splits_tgt  = torch.split(tgt_nk,    tuple(lens_b.tolist()))

        nll_list = []
        for s_nk, t_nk in zip(splits_loss, splits_tgt):
            if head == "categorical":
                mask_nk_b = (t_nk != m.pad)
                w_k = codebook_weights or [1.0] * m.K
                w_k_t = h_ctx_bsd.new_tensor(w_k)  # (K,)
                w_nk = w_k_t.view(1, m.K).expand_as(s_nk) * mask_nk_b.to(s_nk.dtype)
                w_sum_n = w_nk.sum(-1, keepdim=True).clamp_min(1e-8)
                alpha_nk = w_nk / w_sum_n
                nll_pos_n = (s_nk * alpha_nk).sum(-1)
            else:
                mask_nk_b = torch.ones_like(t_nk, dtype=torch.bool)
                denom_n = mask_nk_b.sum(-1).to(s_nk.dtype)
                nll_pos_n = (s_nk.sum(-1) / denom_n)

            valid_n = mask_nk_b.any(-1)
            denom = valid_n.sum().clamp_min(1)
            nll_b = nll_pos_n[valid_n].sum() / denom
            nll_list.append(nll_b)

        return torch.stack(nll_list)


    def forward(
        self,
        tokens_bsc: torch.Tensor,            # (B,S,C)  S=block_size+1
        txt_input_masks_bs: torch.Tensor,    # (B,S)
        img_input_masks_bs: torch.Tensor,    # (B,S)
        aud_input_masks_bs: torch.Tensor,    # (B,S)
        txt_preds_masks_bs_in: Optional[torch.Tensor] = None,  # (B,S-1)
        img_preds_masks_bs_in: Optional[torch.Tensor] = None,
        aud_preds_masks_bs_in: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        return_hidden: bool = False,
        img_codebook_weights: Optional[List[float]] = None,
        aud_codebook_weights: Optional[List[float]] = None,
    ):
        """
        Dimensions notation:
        B = batch size
        S = full sequence length ( = S_in + 1 )
        S_in = input length (S-1)
        C = total codebooks
        K = modality codebooks
        D = backbone hidden size
        V = modality vocab size
        """
        B, S, C = tokens_bsc.shape
        assert C == self.n_codebooks
        assert S <= self.block_size + 1
        S_in = S - 1
        if img_codebook_weights is None:
            img_codebook_weights = self.config.img_codebook_weights
        if aud_codebook_weights is None:
            aud_codebook_weights = self.config.aud_codebook_weights

        if txt_preds_masks_bs_in is not None:
            assert txt_preds_masks_bs_in.shape == (B, S_in)
        if img_preds_masks_bs_in is not None:
            assert img_preds_masks_bs_in.shape == (B, S_in)
        if aud_preds_masks_bs_in is not None:
            assert aud_preds_masks_bs_in.shape == (B, S_in)

        txt_in_mask_bs = txt_input_masks_bs[:, :-1]  # (B,S_in)
        img_in_mask_bs = img_input_masks_bs[:, :-1]
        aud_in_mask_bs = aud_input_masks_bs[:, :-1]

        txt_in_bs  = tokens_bsc[:, :-1, 0] if self.models_txt else None  # (B,S_in)
        txt_tgt_bs = tokens_bsc[:,  1:, 0] if self.models_txt else None  # (B,S_in)

        img_in_bsk = img_tgt_bsk = None
        aud_in_bsk = aud_tgt_bsk = None

        if self.models_img:
            img_in_bsk, img_tgt_bsk = self._prep_modality_io(tokens_bsc, self._mods["img"], sample_bernoulli_inputs=self.training)
        if self.models_aud:
            aud_in_bsk, aud_tgt_bsk = self._prep_modality_io(tokens_bsc, self._mods["aud"], sample_bernoulli_inputs=self.training)

        h_bsd = 0

        if self.models_txt:
            txt_embs_bsd = self.global_workspace(_fsdp_op="embed_text", _tokens=txt_in_bs)  # (B,S_in,D)
            txt_embs_bsd = txt_embs_bsd * txt_in_mask_bs.unsqueeze(-1).to(txt_embs_bsd.dtype)
            h_bsd = h_bsd + txt_embs_bsd

        img_in_hidden = None
        if self.models_img:
            h_img_bsd, img_in_hidden = self._embed_modality(self._mods["img"], img_in_bsk, img_in_mask_bs)
            h_bsd = h_bsd + h_img_bsd

        aud_in_hidden = None
        if self.models_aud:
            h_aud_bsd, aud_in_hidden = self._embed_modality(self._mods["aud"], aud_in_bsk, aud_in_mask_bs)
            h_bsd = h_bsd + h_aud_bsd

        aud_text_prefix_bsd = None
        if self.models_aud and self.aud_expression.codebookt_text_prefix:
            aud_text_prefix_bsd = self.global_workspace(
                _fsdp_op="embed_text", _tokens=txt_tgt_bs
            )

        need_hidden = return_hidden
        if self.models_img and self.img_expression.attention_residual is not None:
            need_hidden = True
        if self.models_aud and self.aud_expression.attention_residual is not None:
            need_hidden = True

        ctx_out = self.global_workspace.context_model(
            inputs_embeds=h_bsd,
            use_cache=False,
            output_hidden_states=need_hidden,
        )

        if return_hidden:
            hidden_lbsd = torch.stack(ctx_out["hidden_states"])[1:]  # (L,B,S_in,D)
            return hidden_lbsd

        h_ctx_bsd = ctx_out["last_hidden_state"]  # (B,S_in,D)
        stacked_h_lbsd = None
        if need_hidden:
            stacked_h_lbsd = torch.stack(ctx_out["hidden_states"])[1:]  # (L,B,S_in,D)

        logits_txt_nv = None
        logits_img_nkv = None
        logits_aud_nkv = None

        txt_loss = h_ctx_bsd.new_zeros(())
        img_loss = h_ctx_bsd.new_zeros(())
        aud_loss = h_ctx_bsd.new_zeros(())
        img_attention_residual_entropy = h_ctx_bsd.new_zeros(())
        aud_attention_residual_entropy = h_ctx_bsd.new_zeros(())

        zero_count = torch.zeros((), dtype=torch.long, device=h_ctx_bsd.device)
        n_txt_preds = txt_preds_masks_bs_in.sum() if txt_preds_masks_bs_in is not None else zero_count
        n_img_preds = img_preds_masks_bs_in.sum() if img_preds_masks_bs_in is not None else zero_count
        n_aud_preds = aud_preds_masks_bs_in.sum() if aud_preds_masks_bs_in is not None else zero_count

        for key, preds_mask_bs_in, in_bsk, tgt_bsk, in_hidden, weights in [
            ("img", img_preds_masks_bs_in, img_in_bsk, img_tgt_bsk, img_in_hidden, img_codebook_weights),
            ("aud", aud_preds_masks_bs_in, aud_in_bsk, aud_tgt_bsk, aud_in_hidden, aud_codebook_weights),
        ]:
            m = self._mods[key]
            if not m.enabled:
                continue
            preds, loss, ent = self._modality_logits_and_loss(
                m, h_ctx_bsd, preds_mask_bs_in,
                in_bsk, tgt_bsk,
                stacked_h_lbsd, in_hidden,
                weights, compute_loss,
                aud_text_prefix_bsd if key == "aud" else None,
            )
            if key == "img":
                logits_img_nkv, img_loss, img_attention_residual_entropy = preds, loss, ent
            else:
                logits_aud_nkv, aud_loss, aud_attention_residual_entropy = preds, loss, ent

        if self.models_txt and txt_preds_masks_bs_in is not None:
            h_txt_nd = h_ctx_bsd[txt_preds_masks_bs_in]
            logits_txt_nv = self.global_workspace(_fsdp_op="unembed_text", _tokens=h_txt_nd)

            if self.swt_token is not None:
                token_ids = torch.arange(logits_txt_nv.size(-1), device=logits_txt_nv.device)
                logits_txt_nv = logits_txt_nv.masked_fill(
                    token_ids == self.swt_token, -float("inf")
                )

            if self.txt_pad_token is not None and not self.config.predict_txt_special_tokens:
                token_ids = torch.arange(logits_txt_nv.size(-1), device=logits_txt_nv.device)
                logits_txt_nv = logits_txt_nv.masked_fill(
                    token_ids >= self.global_workspace.txt_tokenizer.vocab_size,
                    -float("inf"),
                )

            if compute_loss:
                tgt_txt_n = txt_tgt_bs[txt_preds_masks_bs_in]
                if self.config.predict_txt_special_tokens:
                    token_loss = F.cross_entropy(
                        logits_txt_nv, tgt_txt_n, reduction="none"
                    )
                    weights = torch.ones_like(token_loss)
                    if self.txt_pad_token is not None:
                        weights = torch.where(
                            tgt_txt_n == self.txt_pad_token,
                            weights * self.config.txt_pad_loss_weight,
                            weights,
                        )
                    txt_loss = (token_loss * weights).sum() / weights.sum().clamp_min(1)
                else:
                    token_loss = F.cross_entropy(
                        logits_txt_nv,
                        tgt_txt_n,
                        reduction="none",
                        ignore_index=(
                            self.txt_pad_token
                            if self.txt_pad_token is not None
                            else -1
                        ),
                    )
                    valid = (
                        tgt_txt_n != self.txt_pad_token
                        if self.txt_pad_token is not None
                        else torch.ones_like(tgt_txt_n, dtype=torch.bool)
                    )
                    txt_loss = token_loss.sum() / valid.sum().clamp_min(1)
                if not self.training:
                    self.losses["txt"] = txt_loss.detach()

        if compute_loss:
            total_preds = n_txt_preds + n_img_preds + n_aud_preds
            lm_loss = (
                txt_loss * n_txt_preds
                + img_loss * n_img_preds
                + aud_loss * n_aud_preds
            ) / total_preds.clamp_min(1)

            lm_loss = (
                lm_loss
                - self.config.img_attention_residual_entropy_reg
                * img_attention_residual_entropy
                - self.config.aud_attention_residual_entropy_reg
                * aud_attention_residual_entropy
            )

            if not self.training:
                self.losses["lm"] = lm_loss.detach()

            self.last_loss = lm_loss + self.dummy_ops(
                txt_preds_masks_bs_in, img_preds_masks_bs_in, aud_preds_masks_bs_in
            )

        return logits_txt_nv, logits_img_nkv, logits_aud_nkv


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             img_learning_rate=None, aud_learning_rate=None):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        img_lr = learning_rate if img_learning_rate is None else img_learning_rate
        aud_lr = learning_rate if aud_learning_rate is None else aud_learning_rate

        img_names = [pn for pn in param_dict if "img_" in pn]
        aud_names = [pn for pn in param_dict if "aud_" in pn]
        rest_names = [pn for pn in param_dict if pn not in img_names + aud_names]

        def split_decay(names):
            decay = [p for n, p in param_dict.items() if n in names and p.dim() >= 2]
            nodecay = [p for n, p in param_dict.items() if n in names and p.dim() < 2]
            return decay, nodecay

        img_decay, img_nodecay = split_decay(img_names)
        aud_decay, aud_nodecay = split_decay(aud_names)
        rest_decay, rest_nodecay = split_decay(rest_names)

        optim_groups = []
        if img_decay:
            optim_groups.append(
                {"params": img_decay, "weight_decay": weight_decay, "lr": img_lr}
            )
        if img_nodecay:
            optim_groups.append(
                {"params": img_nodecay, "weight_decay": 0.0, "lr": img_lr}
            )
        if aud_decay:
            optim_groups.append(
                {"params": aud_decay, "weight_decay": weight_decay, "lr": aud_lr}
            )
        if aud_nodecay:
            optim_groups.append(
                {"params": aud_nodecay, "weight_decay": 0.0, "lr": aud_lr}
            )
        if rest_decay:
            optim_groups.append(
                {
                    "params": rest_decay,
                    "weight_decay": weight_decay,
                    "lr": learning_rate,
                }
            )
        if rest_nodecay:
            optim_groups.append(
                {"params": rest_nodecay, "weight_decay": 0.0, "lr": learning_rate}
            )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(
        self,
        fwdbwd_per_iter: int,
        dt: float,
        seq_len: Optional[int] = None,
    ):
        """
        Approximate model FLOP utilization (MFU).

        Args:
        fwdbwd_per_iter: number of forward+backward passes per optimizer step
                        (e.g. global_batch_size / micro_batch_size).
        dt: wall-clock seconds per optimizer step.
        seq_len: effective sequence length S_in used during training.
                If None, defaults to self.block_size.
        """
        cfg = self.config
        T = self.block_size if seq_len is None else min(seq_len, self.block_size)
        N = sum(p.numel() for p in self.parameters())
        L_back = self.n_layers
        H_back = self.n_heads
        D_back = self.dim
        Q_back = D_back // H_back
        
        def stack_attn_flops(n_layers: int, dim: int, n_heads: int) -> float:
            if not n_layers or n_layers <= 0:
                return 0.0
            assert dim % n_heads == 0
            q = dim // n_heads
            return 12.0 * n_layers * n_heads * q * T
        
        flops_per_token = 6.0 * N + 12.0 * L_back * H_back * Q_back * T
        extra_attn = 0.0
        
        if self.models_img:
            extra_attn += stack_attn_flops(
                cfg.img_inadapter_n_layers,
                cfg.img_inadapter_dim,
                cfg.img_inadapter_n_heads,
            )
            extra_attn += stack_attn_flops(
                cfg.img_outadapter_n_layers,
                cfg.img_outadapter_dim,
                cfg.img_outadapter_n_heads,
            )

        if self.models_aud:
            extra_attn += stack_attn_flops(
                cfg.aud_inadapter_n_layers,
                cfg.aud_inadapter_dim,
                cfg.aud_inadapter_n_heads,
            )
            extra_attn += stack_attn_flops(
                cfg.aud_outadapter_n_layers,
                cfg.aud_outadapter_dim,
                cfg.aud_outadapter_n_heads,
            )

        for expression, n_codebooks, n_layers, dim, n_heads in (
            (
                self.img_expression if self.models_img else None,
                self.n_img_codebooks,
                cfg.img_codebook_transformer_layers,
                cfg.img_codebook_transformer_dim,
                cfg.img_codebook_transformer_n_heads,
            ),
            (
                self.aud_expression if self.models_aud else None,
                self.n_aud_codebooks,
                cfg.aud_codebook_transformer_layers,
                cfg.aud_codebook_transformer_dim,
                cfg.aud_codebook_transformer_n_heads,
            ),
        ):
            if expression is None or expression.codebookt_layers is None:
                continue
            depth_parameters = sum(
                parameter.numel()
                for parameter in expression.codebookt_layers.parameters()
            )
            flops_per_token += 6.0 * depth_parameters * (n_codebooks - 1)
            flops_per_token += (
                12.0
                * n_layers
                * n_heads
                * (dim // n_heads)
                * n_codebooks
                * n_codebooks
            )

        adapter_factor = 0.25
        flops_per_token += adapter_factor * extra_attn
        flops_per_fwdbwd = flops_per_token * T  # one fwdbwd over a sequence of length T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        device = next(self.parameters()).device
        flops_promised = 312e12  # default A100
        
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device)
            if "H100" in name:
                flops_promised = 989e12
            elif "A100" in name:
                flops_promised = 312e12
        
        return flops_achieved / flops_promised


    def sample_from_logits(self, logits_nv: torch.Tensor, temperature=1.0, top_k=None):
        """
        logits_nv: (N,V)
        returns sampled token indices: (N,1)
        """
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if temperature == 0:
            return logits_nv.argmax(dim=-1, keepdim=True)
        logits_nv = logits_nv / temperature
        if top_k is not None:
            v_nk, _ = torch.topk(logits_nv, min(top_k, logits_nv.size(-1)))
            logits_nv[logits_nv < v_nk[..., [-1]]] = -float("inf")
        probs_nv = F.softmax(logits_nv, dim=-1)
        return torch.multinomial(probs_nv, num_samples=1)

    def _gen_next_modality(
        self,
        m: _ModSpec,
        h_last_b1d: torch.Tensor,           # (1,D)
        stacked_last_lb1d: Optional[torch.Tensor],  # (L,1,1,D) or None
        in_hidden_b1d: Optional[torch.Tensor],
        temperature: float,
        top_k: Optional[int],
        t_rel: int,                         # zero-based generated step
        cache: Optional[List] = None
    ):
        """
        Sample next K tokens for a modality given last hidden.
        Returns nxt_bk: (1,K) integer codes.
        """
        if temperature is None:
            temperature = 1.0
        head = m.expression.head_type
        K = m.K

        h_last_b1d, out_cache, _ = m.expression.forward_with_attention_residual(
            h_last_b1d, stacked_last_lb1d, in_hidden_b1d, use_cache=True, cache=cache)
        if head == "categorical":
            if m.expression.codebookt_layers is None:
                logits_nkv = m.expression.categorical_logits(
                    h_last_b1d.squeeze(1), None
                )
                logits_nkv[..., m.pad] = -float("inf")
                nxt_bk = self.sample_from_logits(
                    logits_nkv.view(-1, logits_nkv.size(-1)),
                    temperature, top_k
                ).view(1, K)
            else:
                h_in_1d = h_last_b1d.squeeze(1)
                if m.expression.out2in_proj is not None:
                    h_in_1d = m.expression.out2in_proj(h_in_1d)
                depth_cache = DynamicCache()
                nxt_bk = torch.empty((1, K), dtype=torch.long, device=h_in_1d.device)
                for codebook in range(K):
                    if codebook == 0:
                        shifted_1d = torch.zeros_like(h_in_1d)
                    else:
                        shifted_1d = m.expression.perception_embed[codebook - 1](
                            nxt_bk[:, codebook - 1]
                        )
                    h_depth_11d = (h_in_1d + shifted_1d).unsqueeze(1)
                    if m.expression.in2codebookt_proj is not None:
                        h_depth_11d = m.expression.in2codebookt_proj(h_depth_11d)
                    position_11 = torch.full(
                        (1, 1), codebook, dtype=torch.long, device=h_in_1d.device
                    )
                    position_embeddings = m.expression.codebookt_rotary(
                        h_depth_11d, position_11
                    )
                    for layer in m.expression.codebookt_layers:
                        h_depth_11d = layer(
                            h_depth_11d,
                            position_ids=position_11,
                            past_key_value=depth_cache,
                            use_cache=True,
                            position_embeddings=position_embeddings,
                        )[0]
                    logits_1v = m.expression.unembed[codebook](
                        h_depth_11d.squeeze(1)
                    )
                    logits_1v[..., m.pad] = -float("inf")
                    nxt_bk[:, codebook] = self.sample_from_logits(
                        logits_1v, temperature, top_k
                    ).squeeze(1)
            for k, d in enumerate(m.delay):
                if t_rel < d:
                    nxt_bk[:, k] = m.pad
            return nxt_bk, out_cache

        if head == "bernoulli":
            logits_bk = m.expression.unembed(h_last_b1d).squeeze(1)
            if temperature == 0:
                bits_bk = logits_bk >= 0
            else:
                probs_bk = torch.sigmoid(logits_bk / temperature)
                bits_bk = torch.rand_like(probs_bk) < probs_bk
            return bits_bk.long() * (m.vocab - 1), out_cache

        if head == "flow":
            sigma = max(temperature, 1e-6)  # avoid zero or negative
            xt_bk = sigma * torch.randn(
                (1, K), device=h_last_b1d.device, dtype=h_last_b1d.dtype
            )  # (1,K) initial noise
            steps = m.expression.flow_steps
            dt = -1.0 / steps
            for i in range(steps):
                tt = 1.0 - float(i) / steps
                t_b = xt_bk.new_full((1,), tt)  # (1,)
                h_ctx_bd = h_last_b1d.squeeze(1)  # (B=1,D)
                v_bk = m.expression.flow_velocity(h_ctx_bd, xt_bk, t_b)  # (1,K)
                xt_bk = xt_bk + v_bk * dt
            x0_bk = xt_bk.clamp(0.0, 1.0)  # (1,K)
            return torch.round(x0_bk * (m.vocab - 1)).long(), out_cache

    @torch.inference_mode()
    def generate(
        self,
        prompt_bsc: torch.Tensor,            # (1,S,C)
        txt_in_mask_bs: torch.Tensor,    # (1,S)
        img_in_mask_bs: torch.Tensor,    # (1,S)
        aud_in_mask_bs: torch.Tensor,    # (1,S)
        max_new_tokens: int,
        gen_txt: Optional[bool] = None,
        gen_img: Optional[bool] = None,
        gen_aud: Optional[bool] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ):
        B, S, C = prompt_bsc.shape
        assert B == 1 and C == self.n_codebooks
        if S == 0:
            raise ValueError("prompt must be non-empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        for name, modality_mask in (
            ("txt", txt_in_mask_bs),
            ("img", img_in_mask_bs),
            ("aud", aud_in_mask_bs),
        ):
            if modality_mask.shape != (B, S):
                raise ValueError(f"{name} input mask must have shape {(B, S)}")
        
        modality_masks = torch.stack([txt_in_mask_bs, img_in_mask_bs, aud_in_mask_bs], dim=0)
        active_per_pos = modality_masks.sum(dim=0)  # (1,S)
        assert (active_per_pos <= 1).all(), "Only one modality can be active at each position"
        enabled = {
            "txt": self.models_txt and gen_txt,
            "img": self.models_img and gen_img,
            "aud": self.models_aud and gen_aud,
        }
        assert sum(bool(v) for v in enabled.values()) == 1, "Exactly one modality must be selected for generation"
        gen_modality = next(k for k, v in enabled.items() if v)
        if (
            gen_modality in self._mods
            and self._mods[gen_modality].expression.codebookt_text_prefix
        ):
            raise ValueError(
                "text-prefixed depth decoding is a pretraining-only configuration"
            )
        if max_new_tokens == 0:
            return prompt_bsc.clone()
        last_idx = S - 1
        last_modality = None
        if txt_in_mask_bs[0, last_idx].item():
            last_modality = "txt"
        elif img_in_mask_bs[0, last_idx].item():
            last_modality = "img"
        elif aud_in_mask_bs[0, last_idx].item():
            last_modality = "aud"
        
        insert_swt = last_modality != gen_modality
        if insert_swt and (not self.models_txt or self.swt_token is None):
            raise ValueError("cross-modal generation requires a configured switch token")
        first_ix_gen = S if insert_swt else S - 1
        generated_modality = self._mods.get(gen_modality)
        delay_tail = 0
        if (
            generated_modality is not None
            and generated_modality.K > 1
            and generated_modality.expression.head_type == "categorical"
        ):
            delay_tail = max(generated_modality.delay)
        max_len = S + max_new_tokens + int(insert_swt) + delay_tail
        if max_len > self.block_size + 1:
            raise ValueError("prompt, switch token, and generation exceed block_size")

        need_hidden = any(
            enabled[k]
            and self._mods[k].expression.attention_residual is not None
            for k in ("img", "aud")
        )

        gen_codes_btc = prompt_bsc.new_full((B, max_len, C), -1)

        if self.models_txt:
            gen_codes_btc[:, :, 0] = self.txt_pad_token
        if self.models_img:
            gen_codes_btc[:, :, self.img_slice] = self.img_pad_token
        if self.models_aud:
            gen_codes_btc[:, :, self.aud_slice] = self.aud_pad_token

        gen_codes_btc[:, :S] = prompt_bsc
        if insert_swt:
            gen_codes_btc[:, first_ix_gen, 0] = self.swt_token

        txt_in_bs  = gen_codes_btc[:, :first_ix_gen, 0] if self.models_txt else None
        txt_in_mask_bs = txt_in_mask_bs[:, :first_ix_gen]
        img_in_bsk = None
        aud_in_bsk = None
        
        if self.models_img:
            img_in_bsk, _ = self._prep_modality_io(gen_codes_btc[:, :first_ix_gen + 1], self._mods["img"])
            img_in_mask_bs = img_in_mask_bs[:, :first_ix_gen]
        if self.models_aud:
            aud_in_bsk, _ = self._prep_modality_io(gen_codes_btc[:, :first_ix_gen + 1], self._mods["aud"])
            aud_in_mask_bs = aud_in_mask_bs[:, :first_ix_gen]
        
        h_bsd = 0

        if self.models_txt:
            txt_embs_bsd = self.global_workspace(_fsdp_op="embed_text", _tokens=txt_in_bs)  # (B,S_in,D)
            txt_embs_bsd = txt_embs_bsd * txt_in_mask_bs.unsqueeze(-1).to(txt_embs_bsd.dtype)
            h_bsd = h_bsd + txt_embs_bsd

        img_in_hidden, img_in_cache = None, None
        if self.models_img:
            h_img_bsd, img_in_hidden = self._embed_modality(
                self._mods["img"], img_in_bsk, img_in_mask_bs
            )
            h_bsd = h_bsd + h_img_bsd

        aud_in_hidden, aud_in_cache = None, None
        if self.models_aud:
            h_aud_bsd, aud_in_hidden = self._embed_modality(
                self._mods["aud"], aud_in_bsk, aud_in_mask_bs
            )
            h_bsd = h_bsd + h_aud_bsd

        if gen_modality in {"img", "aud"} and not insert_swt:
            m = self._mods[gen_modality]
            modality_tokens = img_in_bsk if gen_modality == "img" else aud_in_bsk
            modality_mask = img_in_mask_bs if gen_modality == "img" else aud_in_mask_bs
            suffix = modality_mask.size(1)
            while suffix and bool(modality_mask[0, suffix - 1]):
                suffix -= 1
            if suffix < modality_mask.size(1):
                cache_mask = torch.ones_like(modality_mask[:, suffix:])
                _, _, input_cache = self._embed_modality(
                    m,
                    modality_tokens[:, suffix:],
                    cache_mask,
                    use_cache=True,
                    cache=None,
                )
                if gen_modality == "img":
                    img_in_cache = input_cache
                else:
                    aud_in_cache = input_cache

        if self.models_img and self.img_expression.attention_residual is not None:
            need_hidden = True
        if self.models_aud and self.aud_expression.attention_residual is not None:
            need_hidden = True
        ctx_out = self.global_workspace.context_model(
            inputs_embeds=h_bsd,
            use_cache=True,
            output_hidden_states=need_hidden,
            past_key_values=None,
        )
        backbone_cache = ctx_out["past_key_values"]
        h_ctx_bsd = ctx_out["last_hidden_state"]  # (B,S_in,D)
        stacked_h_lbsd = None
        if need_hidden:
            stacked_h_lbsd = torch.stack(ctx_out["hidden_states"])[1:]  # (L,B,S_in,D)

        img_out_cache = None
        if self.models_img:
            _, img_out_cache, _ = self.img_expression.forward_with_attention_residual(
                h_ctx_bsd, stacked_h_lbsd, img_in_hidden, use_cache=True, cache=None)
        aud_out_cache = None
        if self.models_aud:
            _, aud_out_cache, _ = self.aud_expression.forward_with_attention_residual(
                h_ctx_bsd, stacked_h_lbsd, aud_in_hidden, use_cache=True, cache=None)
        
        in_mask_b1 = torch.ones_like(txt_in_mask_bs[:, [0]])
        for t in range(first_ix_gen, max_len - 1):
            tokens_b1c = gen_codes_btc[:, [t], :]
            img_in_hidden_b1d = None
            aud_in_hidden_b1d = None
            if gen_modality == "txt" or (t == first_ix_gen and insert_swt):
                txt_tokens_b1 = tokens_b1c[..., 0]
                h_curr_b1d = self.global_workspace(_fsdp_op="embed_text", _tokens=txt_tokens_b1)
                if gen_modality in {"img", "aud"}:
                    m = self._mods[gen_modality]
                    _, current_hidden = self._embed_modality(
                        m,
                        tokens_b1c[..., m.sl],
                        torch.zeros_like(in_mask_b1),
                    )
                    if gen_modality == "img":
                        img_in_hidden_b1d = current_hidden
                    else:
                        aud_in_hidden_b1d = current_hidden
            else:
                for k, cache in [
                    ("img", img_in_cache),
                    ("aud", aud_in_cache),
                ]:
                    if not enabled[k]:
                        continue
                    m = self._mods[k]
                    tokens_mod_b1k = tokens_b1c[..., m.sl]
                    if (
                        m.expression.head_type == "categorical"
                        and m.K > 1
                        and t == first_ix_gen
                        and not insert_swt
                    ):
                        tokens_mod_b1k = tokens_mod_b1k.clone()
                        for codebook, delay in enumerate(m.delay):
                            source = t - delay
                            if source >= 0:
                                tokens_mod_b1k[..., codebook] = gen_codes_btc[
                                    :, source, m.sl.start + codebook
                                ]
                            else:
                                tokens_mod_b1k[..., codebook] = m.pad
                    elif m.expression.head_type == "flow":
                        tokens_mod_b1k = self._to_unit(tokens_mod_b1k, m.vocab)
                    elif m.expression.head_type == "bernoulli":
                        thr = 0.5 * float(m.vocab - 1) if m.vocab > 1 else 0.5
                        tokens_mod_b1k = (tokens_mod_b1k > thr).float()
                    m_emb_out = self._embed_modality(
                        m, tokens_mod_b1k, in_mask_b1, use_cache=True, cache=cache
                    )
                    if k == "img":
                        h_curr_b1d, img_in_hidden_b1d, img_in_cache = m_emb_out
                        aud_in_hidden_b1d = None
                    else:  # aud
                        h_curr_b1d, aud_in_hidden_b1d, aud_in_cache = m_emb_out
                        img_in_hidden_b1d = None
            ctx_out = self.global_workspace.context_model(
                inputs_embeds=h_curr_b1d,
                use_cache=True,
                output_hidden_states=need_hidden,
                past_key_values=backbone_cache,
            )
            backbone_cache = ctx_out["past_key_values"]
            h_last_b1d = ctx_out["last_hidden_state"]  # (B,1,D)
            stacked_last_lb1d = None
            if need_hidden:
                stacked_last_lb1d = torch.stack(ctx_out["hidden_states"])[1:]  # (L,B,S_in,D)
            if gen_modality == "txt":
                logits_bv = self.global_workspace(_fsdp_op="unembed_text", _tokens=h_last_b1d.squeeze(1))
                if self.swt_token is not None:
                    logits_bv[..., self.swt_token] = -float("inf")
                if self.txt_pad_token is not None:
                    logits_bv[..., self.txt_pad_token] = -float("inf")
                nxt_b1 = self.sample_from_logits(logits_bv, temperature, top_k).view(1)
                gen_codes_btc[:, t + 1, 0] = nxt_b1
            else:
                for k, in_hidden, cache in [
                    ("img", img_in_hidden_b1d, img_out_cache),
                    ("aud", aud_in_hidden_b1d, aud_out_cache),
                ]:
                    if not enabled[k]:
                        continue
                    m = self._mods[k]
                    nxt_bk, out_cache = self._gen_next_modality(
                        m, h_last_b1d,
                        stacked_last_lb1d, in_hidden,
                        temperature, top_k,
                        (t - first_ix_gen), cache
                    )
                    if (
                        m.expression.head_type == "categorical"
                        and m.K > 1
                        and not insert_swt
                    ):
                        for codebook, delay in enumerate(m.delay):
                            source = t + 1 - delay
                            if 0 <= source < S:
                                nxt_bk[:, codebook] = prompt_bsc[
                                    :, source, m.sl.start + codebook
                                ]
                    gen_codes_btc[:, t + 1, m.sl] = nxt_bk
                    if k == "img":
                        img_out_cache = out_cache
                    else:  # aud
                        aud_out_cache = out_cache
        out_btc = gen_codes_btc.clone()
        for k in ("img", "aud"):
            if not enabled[k]:
                continue
            m = self._mods[k]
            if m.K > 1 and m.expression.head_type == "categorical":
                dmax = max(m.delay)
                unpadded_len = out_btc.size(1) - dmax
                base = m.sl.start
                for kk, d in enumerate(m.delay):
                    out_btc[:, first_ix_gen + 1:unpadded_len, base + kk] = gen_codes_btc[:, first_ix_gen + 1 + d:unpadded_len + d, base + kk]
                out_btc = out_btc[:, :unpadded_len]
        return out_btc


    def _to_unit(self, x_nk: torch.Tensor, vocabsize: int):
        if vocabsize <= 1:
            return x_nk.float()
        return x_nk.float() / float(vocabsize - 1)

    def _flow_nll_per_sample(
        self,
        expr: ExpressionModule,
        h_ctx_bsd: torch.Tensor,
        stacked_h_lbsd: Optional[torch.Tensor],
        stacked_in_lbsd: Optional[torch.Tensor],
        preds_mask_bs: torch.Tensor,
        tgt_bsk: torch.Tensor,
        vocabsize: int,
    ):
        """
        Probability-flow ODE log-likelihood for a joint K-dimensional flow over the patch / codebooks.
        Returns:
          nll_b: (B,) tensor with per-sample NLL, averaged over prediction positions.
        """
        B, S_in, K = tgt_bsk.shape
        steps = expr.flow_steps
        dt = 1.0 / steps

        nll_b = []
        for i in range(B):
            mask_i_s = preds_mask_bs[i:i+1]  # (1, S_in)
            if not mask_i_s.any():
                nll_b.append(h_ctx_bsd.new_zeros(()))
                continue

            tgt_i_sk = tgt_bsk[i:i+1]           # (1, S_in, K)
            tgt_i_nk = tgt_i_sk[mask_i_s]      # (N_i, K)
            if tgt_i_nk.numel() == 0:
                nll_b.append(h_ctx_bsd.new_zeros(()))
                continue

            x0_nk = self._to_unit(tgt_i_nk, vocabsize)     # (N_i, K)
            x_nk = x0_nk.detach().requires_grad_(True)     # (N_i, K)

            logp_n = x_nk.new_zeros((x_nk.size(0),), dtype=x_nk.dtype)  # (N_i,)

            with torch.enable_grad():
                for s in range(steps):
                    t = float(s + 0.5) / steps
                    t_n = x_nk.new_full((x_nk.size(0),), t)  # (N_i,)

                    v_nk, _ = expr(
                        h_ctx_bsd[i:i+1],          # (1, S_in, D)
                        mask_i_s,                  # (1, S_in)
                        None,
                        stacked_h_lbsd=None if stacked_h_lbsd is None else stacked_h_lbsd[:, i:i+1],
                        stacked_in_lbsd=None if stacked_in_lbsd is None else stacked_in_lbsd[:, i:i+1],
                        flow_xt_nk=x_nk,           # (N_i, K)
                        flow_t_n=t_n,              # (N_i,)
                    )

                    eps_nk = torch.randn_like(x_nk)               # (N_i, K)
                    v_dot_eps = (v_nk * eps_nk).sum()             # scalar
                    grad_x = torch.autograd.grad(
                        v_dot_eps,
                        x_nk,
                        create_graph=False,
                        retain_graph=False,
                    )[0]                                         # (N_i, K)
                    div_est_n = (grad_x * eps_nk).sum(-1)         # (N_i,)

                    logp_n = logp_n - div_est_n * dt              # (N_i,)

                    x_nk = (x_nk + v_nk * dt).detach().requires_grad_(True)

            logp_prior_n = -0.5 * (x_nk ** 2).sum(-1) - 0.5 * K * math.log(2 * math.pi)  # (N_i,)
            logp_n = logp_n + logp_prior_n  # (N_i,)

            nll_pos_n = -logp_n
            nll_b.append(nll_pos_n.mean())

        return torch.stack(nll_b)  # (B,)

    def log_likelihood(
        self,
        input_tokens_bsc: torch.Tensor,   # (B,S,C)
        txt_input_masks_bs,
        img_input_masks_bs,
        aud_input_masks_bs,
        txt_preds_masks_bs_in,            # (B,S-1)
        img_preds_masks_bs_in,
        aud_preds_masks_bs_in,
        txt_weight: float = 1.0,
        img_codebook_weights: Optional[List[float]] = None,
        aud_codebook_weights: Optional[List[float]] = None,
    ):
        """
        Returns per-sample log-likelihood:
        ll_b = -(txt_weight * txt_nll_b + img_nll_b + aud_nll_b)

        Aligned with forward:
        - img/aud targets come from FULL delayed streams, then split.
        - per-codebook weighted loss / LL.
        """
        B, S, C = input_tokens_bsc.shape
        assert C == self.n_codebooks

        txt_in_mask_bs = txt_input_masks_bs[:, :-1]
        img_in_mask_bs = img_input_masks_bs[:, :-1]
        aud_in_mask_bs = aud_input_masks_bs[:, :-1]

        txt_in_bs  = input_tokens_bsc[:, :-1, 0] if self.models_txt else None
        txt_tgt_bs = input_tokens_bsc[:,  1:, 0] if self.models_txt else None

        img_in_bsk = img_tgt_bsk = None
        aud_in_bsk = aud_tgt_bsk = None

        if self.models_img:
            img_in_bsk, img_tgt_bsk = self._prep_modality_io(input_tokens_bsc, self._mods["img"])
        if self.models_aud:
            aud_in_bsk, aud_tgt_bsk = self._prep_modality_io(input_tokens_bsc, self._mods["aud"])

        h_bsd = 0
        img_in_hidden = None
        aud_in_hidden = None

        if self.models_txt:
            txt_embs_bsd = self.global_workspace(_fsdp_op="embed_text", _tokens=txt_in_bs)
            txt_embs_bsd = txt_embs_bsd * txt_in_mask_bs.unsqueeze(-1).to(txt_embs_bsd.dtype)
            h_bsd = h_bsd + txt_embs_bsd

        img_in_hidden = None
        if self.models_img:
            h_img_bsd, img_in_hidden = self._embed_modality(self._mods["img"], img_in_bsk, img_in_mask_bs)
            h_bsd = h_bsd + h_img_bsd

        aud_in_hidden = None
        if self.models_aud:
            h_aud_bsd, aud_in_hidden = self._embed_modality(self._mods["aud"], aud_in_bsk, aud_in_mask_bs)
            h_bsd = h_bsd + h_aud_bsd

        aud_text_prefix_bsd = None
        if self.models_aud and self.aud_expression.codebookt_text_prefix:
            aud_text_prefix_bsd = self.global_workspace(
                _fsdp_op="embed_text", _tokens=txt_tgt_bs
            )

        need_hidden = False
        if self.models_img and self.img_expression.attention_residual is not None:
            need_hidden = True
        if self.models_aud and self.aud_expression.attention_residual is not None:
            need_hidden = True

        with torch.no_grad():
            ctx_out = self.global_workspace.context_model(
                inputs_embeds=h_bsd,
                use_cache=False,
                output_hidden_states=need_hidden,
            )

        h_ctx_bsd = ctx_out["last_hidden_state"]  # (B,S_in,D)

        stacked_h_lbsd = None
        if need_hidden:
            stacked_h_lbsd = torch.stack(ctx_out["hidden_states"])[1:]  # (L,B,S_in,D)

        txt_nll_b = input_tokens_bsc.new_zeros((B,), dtype=torch.float)
        img_nll_b = input_tokens_bsc.new_zeros((B,), dtype=torch.float)
        aud_nll_b = input_tokens_bsc.new_zeros((B,), dtype=torch.float)

        if self.models_txt and txt_preds_masks_bs_in is not None and txt_preds_masks_bs_in.any():
            with torch.no_grad():
                h_txt_nd = h_ctx_bsd[txt_preds_masks_bs_in]
                logits_txt_nv = self.global_workspace(_fsdp_op="unembed_text", _tokens=h_txt_nd)
                if not self.config.predict_txt_special_tokens:
                    logits_txt_nv[..., self.global_workspace.txt_tokenizer.vocab_size:] = -float("inf")
                tgt_txt_n = txt_tgt_bs[txt_preds_masks_bs_in]
                per_tok_n = F.cross_entropy(
                    logits_txt_nv,
                    tgt_txt_n,
                    reduction="none",
                    ignore_index=(
                        -1
                        if self.config.predict_txt_special_tokens
                        else self.txt_pad_token
                        if self.txt_pad_token is not None
                        else -1
                    ),
                )
                token_weights_n = torch.ones_like(per_tok_n)
                if self.config.predict_txt_special_tokens and self.txt_pad_token is not None:
                    token_weights_n = torch.where(
                        tgt_txt_n == self.txt_pad_token,
                        token_weights_n * self.config.txt_pad_loss_weight,
                        token_weights_n,
                    )

            lens_b = txt_preds_masks_bs_in.sum(1)
            splits = torch.split(per_tok_n, tuple(lens_b.tolist()))
            weight_splits = torch.split(token_weights_n, tuple(lens_b.tolist()))
            txt_nll_b = torch.stack([
                (losses * weights).sum() / weights.sum().clamp_min(1)
                for losses, weights in zip(splits, weight_splits)
            ])

        for key, preds_mask_bs_in, in_bsk, tgt_bsk, in_hidden, weights in [
            ("img", img_preds_masks_bs_in, img_in_bsk, img_tgt_bsk, img_in_hidden, img_codebook_weights),
            ("aud", aud_preds_masks_bs_in, aud_in_bsk, aud_tgt_bsk, aud_in_hidden, aud_codebook_weights),
        ]:
            m = self._mods[key]
            if not m.enabled:
                continue
            nll_b = self._modality_nll(
                m, h_ctx_bsd, stacked_h_lbsd, in_hidden,
                preds_mask_bs_in, in_bsk, tgt_bsk, weights,
                aud_text_prefix_bsd if key == "aud" else None,
            )
            if key == "img":
                img_nll_b = nll_b
            else:
                aud_nll_b = nll_b

        return -(txt_weight * txt_nll_b + img_nll_b + aud_nll_b)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        """Load a native LF2AR directory without Transformers custom code."""
        root = Path(path_or_repo)
        if not root.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("install huggingface-hub to load a remote model") from exc
            root = Path(snapshot_download(str(path_or_repo)))
        payload = json.loads((root / "config.json").read_text())
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported checkpoint format: {payload.get('format_version')}")
        allowed = {field.name for field in fields(ModelArgs)}
        unknown = set(payload["model_args"]) - allowed
        if unknown:
            raise ValueError(f"unknown model arguments: {sorted(unknown)}")
        raw_args = dict(payload["model_args"])
        tokenizer = raw_args.get("tokenizer")
        if tokenizer and not Path(tokenizer).is_absolute():
            raw_args["tokenizer"] = str(root / tokenizer)
        model = cls(ModelArgs(**raw_args), is_resume=True)
        state = _load_safetensors(root)
        for alias, canonical in payload.get("weight_aliases", {}).items():
            state[alias] = state[canonical]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"incompatible weights: missing={missing}, unexpected={unexpected}")
        model.to(device=device, dtype=dtype) if dtype is not None else model.to(device)
        model.eval()
        return model

    def save_pretrained(self, output_dir: str | Path, *, max_shard_size: int = 4_000_000_000) -> None:
        """Export model-only FP32 safetensors, architecture config, and tokenizer."""
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("install safetensors to export a model") from exc
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if max_shard_size < 1:
            raise ValueError("max_shard_size must be positive")
        if list(output_dir.glob("model*.safetensors*")):
            raise FileExistsError(f"model weights already exist in {output_dir}")

        model_args = asdict(self.config)
        model_args["warm_init"] = False
        model_args["backbone_config"] = self.global_workspace.context_model.config.to_dict()
        if self.global_workspace.txt_tokenizer is not None:
            tokenizer_dir = output_dir / "tokenizer"
            self.global_workspace.txt_tokenizer.save_pretrained(tokenizer_dir)
            model_args["tokenizer"] = "tokenizer"
            model_args["tokenizer_revision"] = None

        state = {}
        aliases = {}
        storage_to_name = {}
        for name, tensor in self.state_dict().items():
            storage = tensor.untyped_storage() if hasattr(tensor, "untyped_storage") else tensor.storage()
            key = (storage.data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
            if key in storage_to_name:
                aliases[name] = storage_to_name[key]
                continue
            storage_to_name[key] = name
            state[name] = tensor.detach().float().cpu().contiguous()

        payload = {
            "format_version": 1,
            "model_type": "interleaved-lm",
            "architectures": [self.__class__.__name__],
            "model_args": model_args,
            "torch_dtype": "float32",
            "weight_aliases": aliases,
        }
        (output_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")

        shards = []
        current, current_size = {}, 0
        for name, tensor in state.items():
            size = tensor.numel() * tensor.element_size()
            if current and current_size + size > max_shard_size:
                shards.append(current)
                current, current_size = {}, 0
            current[name] = tensor
            current_size += size
        if current:
            shards.append(current)
        if len(shards) == 1:
            save_file(shards[0], str(output_dir / "model.safetensors"), metadata={"format": "pt"})
            return
        weight_map = {}
        for index, shard in enumerate(shards, 1):
            filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, str(output_dir / filename), metadata={"format": "pt"})
            weight_map.update({name: filename for name in shard})
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
            "weight_map": weight_map,
        }
        (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def _load_safetensors(root: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("install safetensors to load this model") from exc
    single = root / "model.safetensors"
    index_path = root / "model.safetensors.index.json"
    if single.exists() and index_path.exists():
        raise ValueError(f"ambiguous single-file and sharded weights in {root}")
    if single.exists():
        return load_file(str(single), device="cpu")
    if not index_path.exists():
        raise FileNotFoundError(f"no safetensors weights found in {root}")
    index = json.loads(index_path.read_text())
    if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError("invalid safetensors index")
    state = {}
    for filename in sorted(set(index["weight_map"].values())):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"invalid safetensors shard name: {filename}")
        state.update(load_file(str(root / filename), device="cpu"))
    return state
