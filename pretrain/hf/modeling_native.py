"""This project's rotary models in Hugging Face's format, so that its tools load them.

The layers and parameter names follow pretrain.model's llama blocks: grouped
keys and values, rotary positions, optional per-head normalization of
queries and keys, an optional sigmoid gate on the attention output, SwiGLU,
RMSNorm and input and output embeddings that share their weights. Attention
stays within the documents of a sequence, which begin at document_start, as
in training.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_native import NativeConfig


def rotated(value, cosine, sine):
    first, second = value.chunk(2, dim=-1)
    return value * cosine + torch.cat((-second, first), dim=-1) * sine


class NativeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_width = width // self.heads
        intermediate = config.intermediate_size or math.ceil(8 * width / 3 / 64) * 64
        self.norm_attention = nn.RMSNorm(width, eps=1e-5)
        self.norm_mlp = nn.RMSNorm(width, eps=1e-5)
        self.query = nn.Linear(width, width, bias=False)
        self.key_value = nn.Linear(width, 2 * self.kv_heads * self.head_width, bias=False)
        self.attention_output = nn.Linear(width, width, bias=False)
        self.query_norm = self.key_norm = self.attention_gate = None
        if config.qk_norm:
            self.query_norm = nn.RMSNorm(self.head_width, eps=1e-6)
            self.key_norm = nn.RMSNorm(self.head_width, eps=1e-6)
        if config.attention_gate is not None:
            gates = self.heads if config.attention_gate == "headwise" else width
            self.attention_gate = nn.Linear(width, gates, bias=False)
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, hidden, cosine, sine, mask, cache, index):
        normalized = self.norm_attention(hidden)
        batch, length, width = normalized.shape
        query = self.query(normalized).view(batch, length, self.heads, self.head_width)
        key, value = [
            tensor.view(batch, length, self.kv_heads, self.head_width)
            for tensor in self.key_value(normalized).chunk(2, dim=-1)
        ]
        if self.query_norm is not None:
            query, key = self.query_norm(query), self.key_norm(key)
        query = rotated(query.transpose(1, 2), cosine, sine)
        key = rotated(key.transpose(1, 2), cosine, sine)
        value = value.transpose(1, 2)
        if cache is not None:
            key, value = cache.update(key, value, index)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, enable_gqa=self.heads != self.kv_heads
        ).transpose(1, 2)
        if self.attention_gate is not None:
            gate = torch.sigmoid(self.attention_gate(normalized))
            attended = attended * gate.view(batch, length, self.heads, -1)
        hidden = hidden + self.attention_output(attended.reshape(batch, length, width))
        normalized = self.norm_mlp(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class NativeForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NativeConfig
    base_model_prefix = "native"
    _no_split_modules = ["NativeBlock"]
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(NativeBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def mask(self, input_ids, attention_mask, past, length):
        """Which keys each query sees: earlier positions, not padding, within its document."""
        device = input_ids.device
        queries = torch.arange(past, past + length, device=device)
        keys = torch.arange(past + length, device=device)
        allowed = (keys[None, :] <= queries[:, None])[None, None]
        allowed = allowed.expand(input_ids.shape[0], 1, length, past + length)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, : past + length].bool()
        if self.config.document_start is not None and past == 0:
            documents = (input_ids == self.config.document_start).cumsum(1)
            allowed = allowed & (documents[:, None, :, None] == documents[:, None, None, :])
        # A padding position with nothing to see attends to itself, so that no
        # undefined value reaches the positions that do attend.
        alone = ~allowed.any(-1, keepdim=True) & (keys[None, :] == queries[:, None])[None, None]
        return allowed | alone

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        logits_to_keep=0,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        length = input_ids.shape[1]
        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -length:]
            else:
                position_ids = torch.arange(past, past + length, device=input_ids.device)[None]
        # Computed here rather than kept in a buffer, which loading leaves unset.
        head_width = self.config.hidden_size // self.config.num_attention_heads
        frequencies = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, head_width, 2, device=input_ids.device).float() / head_width)
        )
        angles = position_ids.float()[..., None] * frequencies
        angles = torch.cat((angles, angles), dim=-1)[:, None]
        hidden = self.embedding(input_ids)
        cosine, sine = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
        mask = self.mask(input_ids, attention_mask, past, length)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cosine, sine, mask, past_key_values, index)
        hidden = self.norm(hidden)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        logits = F.linear(hidden, self.embedding.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)
