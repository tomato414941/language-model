"""The configuration of this project's rotary models in Hugging Face's format."""

from transformers import PretrainedConfig


class NativeConfig(PretrainedConfig):
    model_type = "native_rotary"

    def __init__(
        self,
        vocab_size=258,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=None,
        num_hidden_layers=8,
        intermediate_size=None,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        qk_norm=False,
        attention_gate=None,
        document_start=None,
        value_embeddings=False,
        residual_scalars=False,
        tie_word_embeddings=True,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.attention_gate = attention_gate
        self.document_start = document_start
        self.value_embeddings = value_embeddings
        self.residual_scalars = residual_scalars
        self.use_cache = use_cache
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
