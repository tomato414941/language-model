import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    width: int = 512
    heads: int = 8
    layers: int = 8
    context: int = 1024
    architecture: str = "gpt"
    kv_heads: int | None = None
    intermediate_size: int | None = None
    rope_theta: float = 10000.0
    # Scale each head's queries and keys to unit RMS before rotation, as
    # Qwen3 and Gemma 3 do, which steadies training.
    qk_norm: bool = False
    # A sigmoid gate on the attention output, computed from the block's
    # normalized input (Qiu et al., NeurIPS 2025): "headwise" gives each head
    # one gate, "elementwise" one for each of its output dimensions.
    attention_gate: str | None = None
    # The token that begins each document. When set, attention stays within
    # documents, so the documents packed into a window do not see one another.
    document_start: int | None = None
    # One layer in this many attends to every earlier position; the others are
    # Gated DeltaNet layers (Yang et al., ICLR 2025), which read a state of
    # fixed size, as Qwen3-Next and Qwen3.5 put three of them before each
    # attention layer. None keeps attention in every layer.
    full_attention_every: int | None = None
    # A mixture of experts in place of each feed-forward network: each token
    # goes to active_experts of the experts, small SwiGLU networks with
    # expert_size hidden units, by default as many as keep the computation of
    # the dense network.
    experts: int | None = None
    active_experts: int | None = None
    expert_size: int | None = None
    # Each depth predicts one token further ahead, from the depth before it and
    # the embedding of the token in between, as DeepSeek-V3 trains; the
    # predictions add to the training objective only.
    prediction_depth: int = 0

    def __post_init__(self):
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) <= 0
            for name in ("width", "heads", "layers", "context")
        ):
            raise ValueError("model dimensions must be positive integers")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.architecture not in ("gpt", "llama"):
            raise ValueError("architecture must be gpt or llama")
        if self.kv_heads is not None and (
            type(self.kv_heads) is not int
            or self.kv_heads <= 0
            or self.heads % self.kv_heads
        ):
            raise ValueError("kv_heads must divide the query heads")
        if self.intermediate_size is not None and (
            type(self.intermediate_size) is not int or self.intermediate_size <= 0
        ):
            raise ValueError("intermediate_size must be a positive integer")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if self.architecture == "llama" and (self.width // self.heads) % 2:
            raise ValueError("rotary attention requires an even head dimension")
        if type(self.qk_norm) is not bool:
            raise ValueError("qk_norm must be true or false")
        if self.attention_gate not in (None, "headwise", "elementwise"):
            raise ValueError("attention_gate must be headwise or elementwise")
        if self.document_start is not None and (
            type(self.document_start) is not int or self.document_start < 0
        ):
            raise ValueError("document_start must be a token id")
        if self.full_attention_every is not None and (
            type(self.full_attention_every) is not int or self.full_attention_every < 1
        ):
            raise ValueError("full_attention_every must be a positive integer")
        if self.experts is not None:
            if type(self.experts) is not int or self.experts < 2:
                raise ValueError("experts must be an integer of at least 2")
            if (
                type(self.active_experts) is not int
                or not 1 <= self.active_experts <= self.experts
            ):
                raise ValueError("active_experts must be between 1 and experts")
            if self.expert_size is not None and (
                type(self.expert_size) is not int or self.expert_size <= 0
            ):
                raise ValueError("expert_size must be a positive integer")
        elif self.active_experts is not None or self.expert_size is not None:
            raise ValueError("active_experts and expert_size require experts")
        if type(self.prediction_depth) is not int or not 0 <= self.prediction_depth <= 4:
            raise ValueError("prediction_depth must be an integer from 0 to 4")
        if self.architecture == "gpt" and (
            self.kv_heads is not None
            or self.intermediate_size is not None
            or self.qk_norm
            or self.attention_gate is not None
            or self.document_start is not None
            or self.full_attention_every is not None
            or self.experts is not None
            or self.prediction_depth
        ):
            raise ValueError(
                "kv_heads, intermediate_size, qk_norm, attention_gate, document_start, "
                "full_attention_every, experts and prediction_depth require llama architecture"
            )


class Rotary(nn.Module):
    def __init__(self, dimension, theta):
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, dimension, 2).float() / dimension)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, query, key, positions=None):
        if positions is None:
            positions = torch.arange(query.shape[-2], device=query.device)[None]
        angles = positions.float()[..., None] * self.frequencies.float()
        angles = torch.cat((angles, angles), dim=-1)[:, None]
        cosine, sine = angles.cos().to(query.dtype), angles.sin().to(query.dtype)

        def rotate(value):
            first, second = value.chunk(2, dim=-1)
            return value * cosine + torch.cat((-second, first), dim=-1) * sine

        return rotate(query), rotate(key)


# Positions the gated delta rule solves together, and the width of the short
# convolution before it, as Gated DeltaNet's reference implementation uses.
CHUNK = 64
CONVOLUTION = 4
# The log decay at a document's first position, which leaves 1e-13 of the
# state before it: a finite value keeps the chunk's sums of decays exact.
RESET = -30.0


def recurrent_delta_rule(query, key, value, decay, beta, state=None):
    """The gated delta rule one position at a time; its outputs and final state.

    query and key are unit vectors (batch, heads, length, key width) and value
    is (batch, heads, length, value width); decay holds each position's log
    decay and beta its write strength (batch, heads, length). Each head's
    state maps keys to values: every position decays it, corrects what it
    returns for the position's key toward the value by beta, and reads it
    with the query.
    """
    batch, heads, length, width = key.shape
    query, key, value, decay, beta = (x.float() for x in (query, key, value, decay, beta))
    query = query * width**-0.5
    if state is None:
        state = key.new_zeros(batch, heads, width, value.shape[-1])
    outputs = []
    for position in range(length):
        state = state * decay[..., position, None, None].exp()
        written = key[:, :, position]
        correction = beta[..., position, None] * (
            value[:, :, position] - (written[..., None] * state).sum(-2)
        )
        state = state + written[..., None] * correction[..., None, :]
        outputs.append((query[:, :, position, :, None] * state).sum(-2))
    return torch.stack(outputs, 2), state


def chunked_delta_rule(query, key, value, decay, beta, state=None):
    """The gated delta rule computed CHUNK positions at a time; it returns what
    recurrent_delta_rule returns.

    Within a chunk, the corrections solve one triangular system together, in
    the WY form of Gated DeltaNet's chunkwise algorithm; the state then passes
    from chunk to chunk.
    """
    batch, heads, length, width = key.shape
    pad = -length % CHUNK
    query, key, value = (F.pad(x.float(), (0, 0, 0, pad)) for x in (query, key, value))
    decay, beta = (F.pad(x.float(), (0, pad)) for x in (decay, beta))
    chunks = (length + pad) // CHUNK
    query = query.view(batch, heads, chunks, CHUNK, width) * width**-0.5
    key = key.view(batch, heads, chunks, CHUNK, width)
    value = value.view(batch, heads, chunks, CHUNK, -1)
    decay = decay.view(batch, heads, chunks, CHUNK).cumsum(-1)
    beta = beta.view(batch, heads, chunks, CHUNK)
    lower = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=key.device).tril()
    # How much of a write at one position remains at each later one.
    kept = (decay[..., :, None] - decay[..., None, :]).masked_fill(~lower, -math.inf).exp()
    system = torch.eye(CHUNK, device=key.device) + (
        beta[..., None] * kept * (key @ key.transpose(-1, -2))
    ).tril(-1)
    solved = torch.linalg.solve_triangular(
        system,
        torch.cat([beta[..., None] * value, (beta * decay.exp())[..., None] * key], -1),
        upper=False,
        unitriangular=True,
    )
    written, recalled = solved.split([value.shape[-1], width], -1)
    reads = kept * (query @ key.transpose(-1, -2))
    state = key.new_zeros(batch, heads, width, value.shape[-1]) if state is None else state.float()
    outputs = []
    for chunk in range(chunks):
        corrections = written[:, :, chunk] - recalled[:, :, chunk] @ state
        outputs.append(
            (decay[:, :, chunk, :, None].exp() * query[:, :, chunk]) @ state
            + reads[:, :, chunk] @ corrections
        )
        last = decay[:, :, chunk, -1]
        remaining = (last[..., None] - decay[:, :, chunk]).exp()[..., None]
        state = last.exp()[..., None, None] * state + (
            key[:, :, chunk] * remaining
        ).transpose(-1, -2) @ corrections
    outputs = torch.stack(outputs, 2).view(batch, heads, chunks * CHUNK, -1)
    return outputs[:, :, :length], state


class Experts(nn.Module):
    """Route each token to its highest-scoring experts and mix their outputs by score.

    Each expert is a small SwiGLU network. Training also minimizes OLMoE's
    load-balancing loss, which keeps the experts evenly used, and its router
    z-loss, which keeps the router's scores small (arXiv:2409.02060).
    """

    def __init__(self, width, count, active, size):
        super().__init__()
        self.active = active
        self.router = nn.Linear(width, count, bias=False)
        self.gate = nn.Parameter(torch.randn(count, width, size) * 0.02)
        self.up = nn.Parameter(torch.randn(count, width, size) * 0.02)
        self.down = nn.Parameter(torch.randn(count, size, width) * 0.02)

    def forward(self, hidden):
        tokens = hidden.reshape(-1, hidden.shape[-1])
        logits = self.router(tokens).float()
        probabilities = logits.softmax(-1)
        scores, chosen = probabilities.topk(self.active, dim=-1)
        scores = scores / scores.sum(-1, keepdim=True)
        assigned = chosen.flatten()
        order = assigned.argsort(stable=True)
        counts = torch.bincount(assigned, minlength=probabilities.shape[-1])
        sources = order // self.active
        outputs = torch.cat(
            [
                (F.silu(part @ gate) * (part @ up)) @ down
                for part, gate, up, down in zip(
                    tokens[sources].split(counts.tolist()), self.gate, self.up, self.down
                )
            ]
        )
        mixed = tokens.new_zeros(tokens.shape).index_add_(
            0, sources, (outputs * scores.flatten()[order, None]).to(tokens.dtype)
        )
        balance = probabilities.shape[-1] * (
            counts / assigned.numel() * probabilities.mean(0)
        ).sum()
        z_loss = torch.logsumexp(logits, -1).square().mean()
        return mixed.view(hidden.shape), 0.01 * balance + 0.001 * z_loss


class ModernBlock(nn.Module):
    """A pre-norm block: a token mixer, then a feed-forward network or experts.

    It returns the residual stream and the loss that its experts add to the
    training objective.
    """

    def __init__(self, config):
        super().__init__()
        self.norm_attention = nn.RMSNorm(config.width, eps=1e-5)
        self.norm_mlp = nn.RMSNorm(config.width, eps=1e-5)

    def build_feed_forward(self, config):
        intermediate = (
            config.intermediate_size or math.ceil(8 * config.width / 3 / 64) * 64
        )
        if config.experts is None:
            self.gate = nn.Linear(config.width, intermediate, bias=False)
            self.up = nn.Linear(config.width, intermediate, bias=False)
            self.down = nn.Linear(intermediate, config.width, bias=False)
        else:
            size = config.expert_size or max(1, intermediate // config.active_experts)
            self.experts = Experts(config.width, config.experts, config.active_experts, size)

    def forward(self, hidden, positions=None, mask=None, cache=None, starts=None):
        hidden = hidden + self.mix(
            self.norm_attention(hidden), positions, mask, cache, starts
        )
        normalized = self.norm_mlp(hidden)
        if hasattr(self, "experts"):
            update, loss = self.experts(normalized)
            return hidden + update, loss
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized)), 0.0


class LlamaBlock(ModernBlock):
    """Rotary attention over every earlier position, with grouped keys and values."""

    def __init__(self, config):
        super().__init__(config)
        self.heads = config.heads
        self.kv_heads = config.kv_heads or config.heads
        self.head_width = config.width // config.heads
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key_value = nn.Linear(
            config.width, 2 * self.kv_heads * self.head_width, bias=False
        )
        self.attention_output = nn.Linear(config.width, config.width, bias=False)
        self.rotary = Rotary(self.head_width, config.rope_theta)
        self.query_norm = self.key_norm = self.attention_gate = None
        if config.qk_norm:
            self.query_norm = nn.RMSNorm(self.head_width, eps=1e-6)
            self.key_norm = nn.RMSNorm(self.head_width, eps=1e-6)
        if config.attention_gate is not None:
            gates = config.heads if config.attention_gate == "headwise" else config.width
            self.attention_gate = nn.Linear(config.width, gates, bias=False)
        self.build_feed_forward(config)

    def new_cache(self, size):
        return KeyValueCache(size)

    def mix(self, normalized, positions, mask, cache, starts):
        batch, length, width = normalized.shape
        query = self.query(normalized).view(batch, length, self.heads, self.head_width)
        key, value = [
            tensor.view(batch, length, self.kv_heads, self.head_width)
            for tensor in self.key_value(normalized).chunk(2, dim=-1)
        ]
        if self.query_norm is not None:
            query, key = self.query_norm(query), self.key_norm(key)
        query, key = self.rotary(query.transpose(1, 2), key.transpose(1, 2), positions)
        value = value.transpose(1, 2)
        if cache is not None:
            key, value = cache.extend(key, value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            is_causal=mask is None,
            enable_gqa=self.heads != self.kv_heads,
        ).transpose(1, 2)
        if self.attention_gate is not None:
            gate = torch.sigmoid(self.attention_gate(normalized))
            attended = attended * gate.view(batch, length, self.heads, -1)
        return self.attention_output(attended.reshape(batch, length, width))


class DeltaBlock(ModernBlock):
    """Gated DeltaNet (Yang et al., ICLR 2025), laid out as in Qwen3-Next.

    A short convolution mixes each position with the ones before it in its
    document; each head's state then follows the gated delta rule, and a SiLU
    gate scales the normalized result. Its cost grows linearly with length.
    """

    def __init__(self, config):
        super().__init__(config)
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.query_key_value = nn.Linear(config.width, 3 * config.width, bias=False)
        # Initialized as PyTorch initializes a depthwise convolution.
        bound = CONVOLUTION**-0.5
        self.convolution = nn.Parameter(
            torch.empty(3 * config.width, 1, CONVOLUTION).uniform_(-bound, bound)
        )
        self.write = nn.Linear(config.width, config.heads, bias=False)
        self.forget = nn.Linear(config.width, config.heads, bias=False)
        # Decay rates and step sizes start spread out as in Mamba 2, which
        # Gated DeltaNet's reference implementation follows.
        self.decay_rate = nn.Parameter(torch.empty(config.heads).uniform_(1, 16).log())
        step = torch.exp(
            torch.rand(config.heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.decay_step = nn.Parameter(step + torch.log(-torch.expm1(-step)))
        self.output_gate = nn.Linear(config.width, config.width, bias=False)
        self.output_norm = nn.RMSNorm(self.head_width, eps=1e-6)
        self.linear_output = nn.Linear(config.width, config.width, bias=False)
        self.build_feed_forward(config)

    def new_cache(self, size):
        return RecurrentCache()

    def convolve(self, inputs, documents, starts, cache):
        """Mix each position with the CONVOLUTION - 1 before it in its document, then apply SiLU."""
        batch, length, _ = inputs.shape
        history = inputs.new_zeros(batch, CONVOLUTION - 1, inputs.shape[-1])
        earlier = documents.new_full((batch, CONVOLUTION - 1), -1)
        if cache is not None and cache.inputs is not None:
            # The cached inputs belong to the document before this call's first
            # position, or to the same one unless that position begins a new one.
            history = cache.inputs
            first = documents[:, :1] - (starts[:, :1].long() if starts is not None else 0)
            earlier = first.expand(batch, CONVOLUTION - 1)
        padded = torch.cat([history, inputs], 1)
        owners = torch.cat([earlier, documents], 1)
        mixed = 0.0
        for back in range(CONVOLUTION):
            begin = CONVOLUTION - 1 - back
            same = (owners[:, begin : begin + length] == documents)[..., None]
            mixed = mixed + padded[:, begin : begin + length] * same * self.convolution[:, 0, begin]
        if cache is not None:
            last = owners[:, -(CONVOLUTION - 1) :] == documents[:, -1:]
            cache.inputs = padded[:, -(CONVOLUTION - 1) :] * last[..., None]
        return F.silu(mixed)

    def mix(self, normalized, positions, mask, cache, starts):
        batch, length, width = normalized.shape
        documents = (
            starts.long().cumsum(1)
            if starts is not None
            else torch.zeros(batch, length, dtype=torch.long, device=normalized.device)
        )
        mixed = self.convolve(self.query_key_value(normalized), documents, starts, cache)
        query, key, value = (
            part.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
            for part in mixed.chunk(3, dim=-1)
        )
        query, key = F.normalize(query, dim=-1), F.normalize(key, dim=-1)
        beta = torch.sigmoid(self.write(normalized).float()).transpose(1, 2)
        decay = -self.decay_rate.exp() * F.softplus(
            self.forget(normalized).float() + self.decay_step
        )
        decay = decay.transpose(1, 2)
        if starts is not None:
            # A document's first position forgets the state before it.
            decay = decay.masked_fill(starts[:, None, :], RESET)
        rule = recurrent_delta_rule if length < CHUNK else chunked_delta_rule
        outputs, state = rule(
            query, key, value, decay, beta, None if cache is None else cache.state
        )
        if cache is not None:
            cache.state = state
        outputs = self.output_norm(outputs.transpose(1, 2).to(normalized.dtype))
        gate = F.silu(self.output_gate(normalized)).view(batch, length, self.heads, -1)
        return self.linear_output((outputs * gate).reshape(batch, length, width))


class Predictor(nn.Module):
    """One depth of multi-token prediction: a hidden state and the embedding of
    the next token, combined and passed through an attention block."""

    def __init__(self, config):
        super().__init__()
        self.norm_hidden = nn.RMSNorm(config.width, eps=1e-5)
        self.norm_embedding = nn.RMSNorm(config.width, eps=1e-5)
        self.combine = nn.Linear(2 * config.width, config.width, bias=False)
        self.block = LlamaBlock(config)

    def forward(self, hidden, embedded, mask, starts):
        combined = self.combine(
            torch.cat([self.norm_hidden(hidden), self.norm_embedding(embedded)], -1)
        )
        return self.block(combined, mask=mask, starts=starts)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm_attention = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.attention_output = nn.Linear(config.width, config.width, bias=False)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width, bias=False),
        )

    def new_cache(self, size):
        return KeyValueCache(size)

    def forward(self, hidden, positions=None, mask=None, cache=None, starts=None):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(self.norm_attention(hidden)).chunk(3, dim=-1)
        q, k, v = [
            x.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for x in (q, k, v)
        ]
        if cache is not None:
            k, v = cache.extend(k, v)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + self.attention_output(attended)
        return hidden + self.mlp(self.norm_mlp(hidden)), 0.0


class KeyValueCache:
    """Keep one layer's keys and values so later steps process only new positions."""

    def __init__(self, size):
        self.size, self.length, self.keys, self.values = size, 0, None, None

    def extend(self, key, value):
        if self.keys is None:
            shape = (*key.shape[:2], self.size, key.shape[-1])
            self.keys, self.values = key.new_empty(shape), value.new_empty(shape)
        end = self.length + key.shape[2]
        self.keys[:, :, self.length : end] = key
        self.values[:, :, self.length : end] = value
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]


class RecurrentCache:
    """Keep a Gated DeltaNet layer's state and the last inputs to its convolution."""

    def __init__(self):
        self.state, self.inputs = None, None


class LanguageModel(nn.Module):
    def __init__(self, vocabulary_size, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, self.config.width)
        modern = self.config.architecture == "llama"
        self.position = (
            None if modern else nn.Embedding(self.config.context, self.config.width)
        )
        every = self.config.full_attention_every
        self.blocks = nn.Sequential(
            *(
                (Block if not modern else LlamaBlock if every is None or (index + 1) % every == 0 else DeltaBlock)(
                    self.config
                )
                for index in range(self.config.layers)
            )
        )
        self.norm = (
            nn.RMSNorm(self.config.width, eps=1e-5)
            if modern
            else nn.LayerNorm(self.config.width)
        )
        self.predictors = nn.ModuleList(
            Predictor(self.config) for _ in range(self.config.prediction_depth)
        )
        self.apply(self._initialize)
        # Scale residual projections to keep deep, randomly initialized models stable.
        for name, parameter in self.named_parameters():
            if name.endswith(
                (
                    "attention_output.weight",
                    "mlp.2.weight",
                    "down.weight",
                    "experts.down",
                    "linear_output.weight",
                )
            ):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * self.config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def document_mask(self, starts):
        """Which earlier positions each position sees within its own document.

        None when documents are not separated, for plain causal attention.
        """
        if starts is None:
            return None
        documents = starts.cumsum(1)
        length = starts.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=starts.device).tril()
        return ((documents[:, :, None] == documents[:, None, :]) & causal)[:, None]

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("input must be a nonempty batch within the context length")
        hidden = self.embedding(tokens)
        if self.position is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            hidden = hidden + self.position(positions)
        starts = (
            None if self.config.document_start is None else tokens == self.config.document_start
        )
        mask = self.document_mask(starts)
        objective = 0.0
        for block in self.blocks:
            hidden, extra = block(hidden, mask=mask, starts=starts)
            objective = objective + extra
        # Input and output embeddings share the same learned weights.
        logits = F.linear(self.norm(hidden), self.embedding.weight)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if not self.training:
            return loss
        predictions = []
        for depth, predictor in enumerate(self.predictors, 1):
            span = tokens.shape[1] - depth
            if span <= 0:
                break
            hidden, extra = predictor(
                hidden[:, :span],
                self.embedding(tokens[:, depth:]),
                None if mask is None else mask[..., :span, :span],
                None if starts is None else starts[:, :span],
            )
            objective = objective + extra
            logits = F.linear(self.norm(hidden), self.embedding.weight)
            predictions.append(
                F.cross_entropy(logits.flatten(0, 1), targets[:, depth:].flatten())
            )
        if predictions:
            # DeepSeek-V3 weighs the further predictions together by 0.3.
            loss = loss + 0.3 * sum(predictions) / len(predictions)
        return loss + objective

    @torch.inference_mode()
    def generate(
        self,
        tokenizer,
        prompt="",
        max_new_tokens=128,
        temperature=0.8,
        top_k=50,
        seed=42,
        top_p=1.0,
    ):
        generated = self.sample_tokens(
            tokenizer, prompt, max_new_tokens, temperature, top_k, seed, top_p
        )
        eos = tokenizer.token_to_id("<|eos|>")
        return prompt + tokenizer.decode(
            [token for token in generated if token != eos], skip_special_tokens=True
        )

    def sample_tokens(
        self,
        tokenizer,
        prompt="",
        max_new_tokens=128,
        temperature=0.8,
        top_k=50,
        seed=42,
        top_p=1.0,
        stop_requested=None,
    ):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be finite and within (0, 1]")
        device = self.embedding.weight.device
        bos = tokenizer.token_to_id("<|bos|>")
        eos = tokenizer.token_to_id("<|eos|>")
        tokens = [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids]
        generator = torch.Generator(device=device).manual_seed(seed)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                if stop_requested is not None and stop_requested():
                    break
                window = torch.tensor([tokens[-self.config.context :]], device=device)
                # Keep inference mode local to each step when a caller streams tokens.
                with torch.inference_mode():
                    logits = self(window)[0, -1].float()
                    logits[bos] = -torch.inf
                    if temperature == 0:
                        token = logits.argmax().item()
                    else:
                        logits = logits / temperature
                        if top_k:
                            cutoff = logits.topk(min(top_k, logits.numel())).values[-1]
                            logits = logits.masked_fill(logits < cutoff, -torch.inf)
                        if top_p < 1:
                            ordered, indices = logits.sort(descending=True)
                            excluded = ordered.softmax(-1).cumsum(-1) > top_p
                            excluded[1:] = excluded[:-1].clone()
                            excluded[0] = False
                            logits[indices[excluded]] = -torch.inf
                        token = torch.multinomial(
                            logits.softmax(-1), 1, generator=generator
                        ).item()
                yield token
                if token == eos:
                    break
                tokens.append(token)
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def continue_greedily(self, sequences, max_new_tokens, end, banned=(), finished=None):
        """Return each sequence's greedy continuation before `end`, decoding all at once.

        A sequence also stops once `finished(row, continuation)` is true of it.
        """
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not sequences or not all(sequences):
            raise ValueError("provide at least one nonempty token sequence")
        longest = max(map(len, sequences))
        # The input of the last step occupies the final cached position.
        size = longest + max_new_tokens - 1
        if size > self.config.context:
            raise ValueError(
                "sequences and new tokens must fit within the context length"
            )
        device = self.embedding.weight.device
        padding = torch.tensor(
            [longest - len(sequence) for sequence in sequences], device=device
        )
        # Shorter sequences are padded on the left with their first token, then masked.
        tokens = torch.tensor(
            [
                [sequence[0]] * (longest - len(sequence)) + sequence
                for sequence in sequences
            ],
            device=device,
        )
        banned = torch.tensor(list(banned), dtype=torch.long, device=device)
        steps = torch.arange(size, device=device)
        real = steps >= padding[:, None]
        caches = [block.new_cache(size) for block in self.blocks]
        # Each row's first real position begins its document, so the padding
        # before it leaves nothing in the recurrent layers' states.
        starts = steps[None, :longest] == padding[:, None]
        continuations = [[] for _ in sequences]
        active = [True] * len(sequences)
        start = 0
        was_training = self.training
        self.eval()
        try:
            for step in range(max_new_tokens):
                stop = start + tokens.shape[1]
                queries, keys = steps[start:stop], steps[:stop]
                # Padding positions attend to themselves so that every row stays finite.
                mask = ((keys <= queries[:, None]) & real[:, None, :stop]) | (
                    keys == queries[:, None]
                )
                positions = (queries - padding[:, None]).clamp(min=0)
                hidden = self.embedding(tokens)
                if self.position is not None:
                    hidden = hidden + self.position(positions)
                for block, cache in zip(self.blocks, caches):
                    hidden, _ = block(
                        hidden, positions, mask[:, None], cache, starts if step == 0 else None
                    )
                logits = F.linear(self.norm(hidden[:, -1]), self.embedding.weight)
                logits = logits.float()
                logits[:, banned] = -torch.inf
                chosen = logits.argmax(-1)
                for row, token in enumerate(chosen.tolist()):
                    if active[row]:
                        if token == end:
                            active[row] = False
                        else:
                            continuations[row].append(token)
                            if finished is not None and finished(row, continuations[row]):
                                active[row] = False
                if not any(active) or step == max_new_tokens - 1:
                    break
                tokens, start = chosen[:, None], stop
        finally:
            self.train(was_training)
        return continuations
