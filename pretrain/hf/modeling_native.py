"""This project's rotary models in Hugging Face's format, so that its tools load them.

The layers and parameter names follow pretrain.model's llama blocks: grouped
keys and values, rotary positions, optional per-head normalization of
queries and keys, an optional sigmoid gate on the attention output, SwiGLU or
a squared ReLU, or a mixture of such networks as experts, RMSNorm and input
and output embeddings that share their weights. Optional value embeddings in every other layer add to the attention's
values through a gate per key-value head, and optional learned scalars scale
the stream and mix the first layer's input back in before each layer.
Attention stays within the documents of a sequence, which begin at
document_start, as in training. A model that reads a count table (COUNT,
Magnusson et al., arXiv 2610.20456) adds to each token's embedding the
share-weighted embeddings of the tokens counted most often after its last few
tokens; the table's arrays are kept with the weights.

NativeModel holds the layers up to the final norm and is what AutoModel loads;
NativeForCausalLM reads the logits from it through the shared embedding.
Attention other than eager and SDPA goes through Transformers' attention
interface, which is how vLLM's Transformers backend runs the model: there each
sequence it is handed is one document, so documents within a row are not
separated.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_native import NativeConfig

# The channels of a block's normalized input that set its value embedding's gate.
VALUE_GATE_CHANNELS = 12
# How a count table keys a history, as pretrain.count does: two polynomial
# hashes modulo primes below 2**31, of which the top 48 bits are kept.
COUNT_PRIMES = (2_147_483_647, 2_147_483_629)
COUNT_BASES = (1_000_003, 998_244_353)
COUNT_KEY_SHIFT = 14
# The most tokens before a position that its lookups read.
COUNT_HISTORY = 8
# The attention this file computes itself, with its own mask.
OWN_ATTENTION = (None, "eager", "sdpa")


def history_keys(tokens, orders):
    """The key of the history of each length from 1 to orders - 1 ending at each position."""
    following = tokens + 1
    first = second = following * 0
    keys = []
    for _ in range(1, orders):
        first = (F.pad(first[..., :-1], (1, 0)) * COUNT_BASES[0] + following) % COUNT_PRIMES[0]
        second = (F.pad(second[..., :-1], (1, 0)) * COUNT_BASES[1] + following) % COUNT_PRIMES[1]
        keys.append(((first << 31) | second) >> COUNT_KEY_SHIFT)
    return keys


def rotated(value, cosine, sine):
    first, second = value.chunk(2, dim=-1)
    return value * cosine + torch.cat((-second, first), dim=-1) * sine


def has_value_embedding(config, index):
    """Whether the layer adds a value embedding: every other layer, the last among them."""
    return config.value_embeddings and index % 2 == (config.num_hidden_layers - 1) % 2


class NativeExperts(nn.Module):
    """A mixture of experts in place of a feed-forward network, as pretrain.model's Experts.

    Each token goes to its highest-scoring experts, with each expert's bias
    added to the scores when the model learned one, and mixes their outputs by
    its scores; a shared expert adds its output for every token. On a GPU in
    bfloat16 the experts multiply in grouped products, without waiting for the
    device, so that vLLM can capture the layer; elsewhere one by one.
    """

    def __init__(self, config):
        super().__init__()
        width, count, size = config.hidden_size, config.experts, config.expert_size
        self.active = config.active_experts
        self.router = nn.Linear(width, count, bias=False)
        squared = config.feed_forward == "relu_squared"
        if squared:
            self.register_parameter("gate", None)
        else:
            self.gate = nn.Parameter(torch.zeros(count, width, size))
        self.up = nn.Parameter(torch.zeros(count, width, size))
        self.down = nn.Parameter(torch.zeros(count, size, width))
        self.shared = None
        if config.shared_expert_size:
            shared = config.shared_expert_size
            self.shared = nn.ModuleDict({
                **({} if squared else {"gate": nn.Linear(width, shared, bias=False)}),
                "up": nn.Linear(width, shared, bias=False),
                "down": nn.Linear(shared, width, bias=False),
            })
        self.balanced_by_bias = config.expert_balance == "bias"
        if self.balanced_by_bias:
            self.register_buffer("bias", torch.zeros(count))

    @staticmethod
    def network(inputs, gate, up, down, product):
        units = product(inputs, up)
        units = F.silu(product(inputs, gate)) * units if gate is not None else F.relu(units).square()
        return product(units, down)

    def forward(self, hidden):
        tokens = hidden.reshape(-1, hidden.shape[-1])
        probabilities = self.router(tokens).float().softmax(-1)
        if self.balanced_by_bias:
            chosen = (probabilities + self.bias).topk(self.active, dim=-1).indices
            scores = probabilities.gather(-1, chosen)
        else:
            scores, chosen = probabilities.topk(self.active, dim=-1)
        scores = scores / scores.sum(-1, keepdim=True)
        assigned = chosen.flatten()
        order = assigned.argsort(stable=True)
        counts = torch.bincount(assigned, minlength=probabilities.shape[-1])
        sources = order // self.active
        selected = tokens[sources]
        grouped = getattr(F, "grouped_mm", None)
        if tokens.is_cuda and tokens.dtype == torch.bfloat16 and grouped is not None:
            ends = counts.cumsum(0, dtype=torch.int32)
            outputs = self.network(selected, self.gate, self.up, self.down, lambda a, b: grouped(a, b, offs=ends))
        else:
            outputs = torch.cat([
                self.network(part, None if self.gate is None else self.gate[index],
                             self.up[index], self.down[index], torch.matmul)
                for index, part in enumerate(selected.split(counts.tolist()))
            ])
        mixed = tokens.new_zeros(tokens.shape).index_add_(
            0, sources, (outputs * scores.flatten()[order, None]).to(tokens.dtype)
        )
        if self.shared is not None:
            shared = self.shared
            mixed = mixed + self.network(
                tokens, shared["gate"] if "gate" in shared else None, shared["up"], shared["down"],
                lambda inputs, linear: linear(inputs),
            )
        return mixed.view(hidden.shape)


class NativeBlock(nn.Module):
    def __init__(self, config, layer_idx, value_embedding=False):
        super().__init__()
        self.config = config
        # Read by Transformers' attention interface and by vLLM to find this layer's cache.
        self.layer_idx = layer_idx
        self.is_causal = True
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.heads // self.kv_heads
        self.head_width = width // self.heads
        squared = config.feed_forward == "relu_squared"
        intermediate = config.intermediate_size or (
            4 * width if squared else math.ceil(8 * width / 3 / 64) * 64
        )
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
        self.value_gate = (
            nn.Linear(VALUE_GATE_CHANNELS, self.kv_heads, bias=False) if value_embedding else None
        )
        self.experts = NativeExperts(config) if config.experts else None
        if self.experts is None:
            self.gate = None if squared else nn.Linear(width, intermediate, bias=False)
            self.up = nn.Linear(width, intermediate, bias=False)
            self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, hidden, cosine, sine, mask, cache, values=None, **kwargs):
        normalized = self.norm_attention(hidden)
        batch, length, width = normalized.shape
        query = self.query(normalized).view(batch, length, self.heads, self.head_width)
        key, value = [
            tensor.view(batch, length, self.kv_heads, self.head_width)
            for tensor in self.key_value(normalized).chunk(2, dim=-1)
        ]
        if self.value_gate is not None:
            # Between 0 and 3 times each head's embedded values.
            gate = 3 * torch.sigmoid(self.value_gate(normalized[..., :VALUE_GATE_CHANNELS]))
            value = value + gate[..., None] * values.view(batch, length, self.kv_heads, self.head_width)
        if self.query_norm is not None:
            query, key = self.query_norm(query), self.key_norm(key)
        query = rotated(query.transpose(1, 2), cosine, sine)
        key = rotated(key.transpose(1, 2), cosine, sine)
        value = value.transpose(1, 2)
        if cache is not None:
            key, value = cache.update(key, value, self.layer_idx)
        implementation = self.config._attn_implementation
        if implementation in OWN_ATTENTION:
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, enable_gqa=self.heads != self.kv_heads
            ).transpose(1, 2)
        else:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            attended, _ = ALL_ATTENTION_FUNCTIONS[implementation](
                self, query, key, value, None, scaling=self.head_width**-0.5, dropout=0.0, **kwargs
            )
        attended = attended.reshape(batch, length, self.heads, self.head_width)
        if self.attention_gate is not None:
            gate = torch.sigmoid(self.attention_gate(normalized))
            attended = attended * gate.view(batch, length, self.heads, -1)
        hidden = hidden + self.attention_output(attended.reshape(batch, length, width))
        normalized = self.norm_mlp(hidden)
        if self.experts is not None:
            return hidden + self.experts(normalized)
        if self.gate is None:
            return hidden + self.down(F.relu(self.up(normalized)).square())
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class NativePreTrainedModel(PreTrainedModel):
    config_class = NativeConfig
    base_model_prefix = "model"
    _no_split_modules = ["NativeBlock"]
    _supports_sdpa = True
    # The blocks take any attention through Transformers' attention interface.
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)


class NativeModel(NativePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            NativeBlock(config, index, has_value_embedding(config, index))
            for index in range(config.num_hidden_layers)
        )
        kv_width = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        self.value_embeddings = nn.ModuleDict(
            {
                str(index): nn.Embedding(config.vocab_size, kv_width)
                for index in range(config.num_hidden_layers)
                if has_value_embedding(config, index)
            }
        )
        if config.residual_scalars:
            self.residual_scales = nn.Parameter(torch.ones(config.num_hidden_layers))
            self.input_scales = nn.Parameter(torch.zeros(config.num_hidden_layers))
        if config.count_table:
            table = config.count_table
            token = getattr(torch, table["candidates_dtype"])
            width = table["candidates"]
            self.register_buffer("count_candidates_0", torch.zeros(width, dtype=token))
            self.register_buffer("count_shares_0", torch.zeros(width, dtype=torch.float16))
            for length, rows in enumerate(table["rows"], 1):
                self.register_buffer(f"count_keys_{length}", torch.zeros(rows, dtype=torch.long))
                self.register_buffer(f"count_candidates_{length}", torch.zeros(rows, width, dtype=token))
                self.register_buffer(f"count_shares_{length}", torch.zeros(rows, width, dtype=torch.float16))
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.post_init()

    def counted(self, input_ids, history=None):
        """The share-weighted embeddings of each position's counted continuations.

        history holds the tokens before input_ids that the lookups read, as a
        cached generation step's input lacks them.
        """
        table = self.config.count_table
        read = input_ids if history is None else torch.cat([history, input_ids], 1)
        positions = torch.arange(read.shape[1], device=read.device)
        begins = torch.cummax(torch.where(read == table["document_start"], positions, -1), dim=-1).values
        shape = (*read.shape, table["candidates"])
        ids, shares = [self.count_candidates_0.expand(shape)], [self.count_shares_0.expand(shape)]
        for length, key in enumerate(history_keys(read, table["orders"]), 1):
            keys = getattr(self, f"count_keys_{length}")
            if not len(keys):
                continue
            row = torch.searchsorted(keys, key).clamp_(max=len(keys) - 1)
            earliest = positions - length + 1
            found = (earliest >= 0) & (begins <= earliest) & (keys[row] == key)
            ids.append(getattr(self, f"count_candidates_{length}")[row])
            shares.append(getattr(self, f"count_shares_{length}")[row] * found.unsqueeze(-1))
        new = input_ids.shape[1]
        ids = torch.cat(ids, -1)[:, -new:]
        # Each order's row counts once, an order without a row as nothing.
        shares = torch.cat(shares, -1)[:, -new:] / table["orders"]
        weight = self.embedding.weight
        gathered = F.embedding_bag(
            ids.flatten(0, 1).long(), weight, per_sample_weights=shares.flatten(0, 1).to(weight.dtype), mode="sum"
        )
        return gathered.view(*input_ids.shape, -1)

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
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        count_history=None,
        return_dict=None,
        **kwargs,
    ):
        if input_ids is None and (self.config.value_embeddings or self.config.count_table):
            raise ValueError("value embeddings and count tables read the token ids")
        if self.config.count_table and self.config._attn_implementation not in OWN_ATTENTION:
            # vLLM hands each step only its new tokens, so the lookups would lose the history before them.
            raise ValueError("a model that reads a count table runs only with its own attention")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        length = hidden.shape[1]
        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -length:]
            else:
                position_ids = torch.arange(past, past + length, device=hidden.device)[None]
        # Computed here rather than kept in a buffer, which loading leaves unset.
        head_width = self.config.hidden_size // self.config.num_attention_heads
        frequencies = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, head_width, 2, device=hidden.device).float() / head_width)
        )
        angles = position_ids.float()[..., None] * frequencies
        angles = torch.cat((angles, angles), dim=-1)[:, None]
        if self.config.count_table:
            hidden = hidden + self.counted(input_ids, count_history)
        cosine, sine = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
        mask = None
        if self.config._attn_implementation in OWN_ATTENTION:
            if input_ids is None:
                raise ValueError("the documents' attention mask reads the token ids")
            mask = self.mask(input_ids, attention_mask, past, length)
        first = hidden
        for index, block in enumerate(self.blocks):
            if self.config.residual_scalars:
                hidden = self.residual_scales[index] * hidden + self.input_scales[index] * first
            table = self.value_embeddings[str(index)] if str(index) in self.value_embeddings else None
            values = None if table is None else table(input_ids)
            hidden = block(hidden, cosine, sine, mask, past_key_values, values, **kwargs)
        hidden = self.norm(hidden)
        if return_dict is False:
            return tuple(part for part in (hidden, past_key_values) if part is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


class NativeForCausalLM(NativePreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = NativeModel(config)
        self.post_init()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        cached = input_ids.shape[1] - inputs["input_ids"].shape[1]
        if self.config.count_table and cached:
            inputs["count_history"] = input_ids[:, max(cached - COUNT_HISTORY, 0) : cached]
        return inputs

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        labels=None,
        logits_to_keep=0,
        count_history=None,
        **kwargs,
    ):
        # Generation asks for a dictionary, which this forward always reads from the layers.
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            count_history=count_history,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        logits = F.linear(hidden, self.model.embedding.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)
