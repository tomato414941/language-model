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
        if self.architecture == "gpt" and (
            self.kv_heads is not None or self.intermediate_size is not None
        ):
            raise ValueError(
                "kv_heads and intermediate_size require llama architecture"
            )


class Rotary(nn.Module):
    def __init__(self, dimension, theta):
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, dimension, 2).float() / dimension)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, query, key):
        positions = torch.arange(query.shape[-2], device=query.device).float()
        angles = torch.outer(positions, self.frequencies.float())
        angles = torch.cat((angles, angles), dim=-1)[None, None]
        cosine, sine = angles.cos().to(query.dtype), angles.sin().to(query.dtype)

        def rotate(value):
            first, second = value.chunk(2, dim=-1)
            return value * cosine + torch.cat((-second, first), dim=-1) * sine

        return rotate(query), rotate(key)


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.kv_heads or config.heads
        self.head_width = config.width // config.heads
        intermediate = (
            config.intermediate_size or math.ceil(8 * config.width / 3 / 64) * 64
        )
        self.norm_attention = nn.RMSNorm(config.width, eps=1e-5)
        self.norm_mlp = nn.RMSNorm(config.width, eps=1e-5)
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.key_value = nn.Linear(
            config.width, 2 * self.kv_heads * self.head_width, bias=False
        )
        self.attention_output = nn.Linear(config.width, config.width, bias=False)
        self.rotary = Rotary(self.head_width, config.rope_theta)
        self.gate = nn.Linear(config.width, intermediate, bias=False)
        self.up = nn.Linear(config.width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, config.width, bias=False)

    def forward(self, hidden):
        batch, length, width = hidden.shape
        normalized = self.norm_attention(hidden)
        query = self.query(normalized).view(batch, length, self.heads, self.head_width)
        key, value = self.key_value(normalized).chunk(2, dim=-1)
        key, value = [
            tensor.view(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
            for tensor in (key, value)
        ]
        query, key = self.rotary(query.transpose(1, 2), key)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=self.heads != self.kv_heads
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + self.attention_output(attended)
        normalized = self.norm_mlp(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


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

    def forward(self, hidden):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(self.norm_attention(hidden)).chunk(3, dim=-1)
        q, k, v = [
            x.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for x in (q, k, v)
        ]
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + self.attention_output(attended)
        return hidden + self.mlp(self.norm_mlp(hidden))


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
        block = LlamaBlock if modern else Block
        self.blocks = nn.Sequential(
            *(block(self.config) for _ in range(self.config.layers))
        )
        self.norm = (
            nn.RMSNorm(self.config.width, eps=1e-5)
            if modern
            else nn.LayerNorm(self.config.width)
        )
        self.apply(self._initialize)
        # Scale residual projections to keep deep, randomly initialized models stable.
        for name, parameter in self.named_parameters():
            if name.endswith(
                ("attention_output.weight", "mlp.2.weight", "down.weight")
            ):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * self.config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("input must be a nonempty batch within the context length")
        hidden = self.embedding(tokens)
        if self.position is not None:
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            hidden = hidden + self.position(positions)
        # Input and output embeddings share the same learned weights.
        logits = F.linear(self.norm(self.blocks(hidden)), self.embedding.weight)
        if targets is None:
            return logits
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    @torch.inference_mode()
    def generate(
        self,
        tokenizer,
        prompt="",
        max_new_tokens=128,
        temperature=0.8,
        top_k=50,
        seed=42,
    ):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        device = self.embedding.weight.device
        bos = tokenizer.token_to_id("<|bos|>")
        eos = tokenizer.token_to_id("<|eos|>")
        tokens = [bos, *tokenizer.encode(prompt, add_special_tokens=False).ids]
        generated = []
        generator = torch.Generator(device=device).manual_seed(seed)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                window = torch.tensor([tokens[-self.config.context :]], device=device)
                logits = self(window)[0, -1].float()
                logits[bos] = -torch.inf
                if temperature == 0:
                    token = logits.argmax().item()
                else:
                    logits = logits / temperature
                    if top_k:
                        cutoff = logits.topk(min(top_k, logits.numel())).values[-1]
                        logits = logits.masked_fill(logits < cutoff, -torch.inf)
                    token = torch.multinomial(
                        logits.softmax(-1), 1, generator=generator
                    ).item()
                if token == eos:
                    break
                tokens.append(token)
                generated.append(token)
        finally:
            self.train(was_training)
        return prompt + tokenizer.decode(generated, skip_special_tokens=True)
