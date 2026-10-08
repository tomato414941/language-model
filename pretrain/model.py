import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    width: int = 512
    heads: int = 8
    layers: int = 8
    context: int = 1024

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in asdict(self).values()):
            raise ValueError("model dimensions must be positive integers")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


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
        self.position = nn.Embedding(self.config.context, self.config.width)
        self.blocks = nn.Sequential(
            *(Block(self.config) for _ in range(self.config.layers))
        )
        self.norm = nn.LayerNorm(self.config.width)
        self.apply(self._initialize)
        # Scale residual projections to keep deep, randomly initialized models stable.
        for name, parameter in self.named_parameters():
            if name.endswith(("attention_output.weight", "mlp.2.weight")):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * self.config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("input must be a nonempty batch within the context length")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
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
