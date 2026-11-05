"""Write a checkpoint in Hugging Face's format, with the code that loads it.

Hugging Face's tools then load the model as they load any other, through
AutoModelForCausalLM with trust_remote_code. The tokenizer puts BOS before
each text, as training did, and its chat template writes BOS and then the
conversation format that pretrain.conversation defines: instruction tuning
began each conversation with BOS, and servers such as vLLM tokenize a rendered
conversation without adding special tokens.
"""

import dataclasses
import json
from pathlib import Path

from tokenizers import Tokenizer, processors

from .conversation import MARKERS, THINK_END, THINK_START, TURN_END, TURN_START
from .data import BOS, EOS
from .model import ModelConfig
from .training import load_checkpoint, saved_count_table

# The settings of pretrain.model that the Hugging Face model reproduces, with
# the two that shape only training: multi-token prediction, which the export
# leaves out, and the starting weights. A model that sets any other away from
# its default is refused, so that a part added to pretrain.model cannot go
# missing from the export unnoticed.
SUPPORTED = {
    "width", "heads", "layers", "context", "architecture", "kv_heads", "intermediate_size",
    "rope_theta", "qk_norm", "attention_gate", "document_start", "value_embeddings",
    "residual_scalars", "prediction_depth", "zero_output_init", "feed_forward", "count_table",
}
CHAT_TEMPLATE = (
    f"{BOS}"
    "{%- for message in messages -%}"
    f"{TURN_START}{{{{ message['role'] }}}}\n"
    f"{{% if message['role'] == 'assistant' %}}{THINK_START}{THINK_END}{{% endif %}}"
    f"{{{{ message['content'] }}}}{TURN_END}"
    "{%- endfor -%}"
    f"{{%- if add_generation_prompt %}}{TURN_START}assistant\n{THINK_START}{THINK_END}{{%- endif %}}"
)


def native_config(config, tokenizer, table=None):
    """The Hugging Face configuration of a model, and of the count table it reads, if any."""
    from .hf.configuration_native import NativeConfig

    if config.architecture != "llama":
        raise ValueError("only rotary (llama) models have a Hugging Face form")
    if config.count_table and table is None:
        raise ValueError("the model reads a count table, which the checkpoint does not carry")
    unsupported = [
        field.name
        for field in dataclasses.fields(config)
        if field.name not in SUPPORTED and getattr(config, field.name) != field.default
    ]
    if unsupported:
        raise ValueError(f"the Hugging Face model lacks {', '.join(unsupported)}")
    markers = all(tokenizer.token_to_id(marker) is not None for marker in MARKERS)
    ends = [tokenizer.token_to_id(EOS)]
    if markers:
        ends.insert(0, tokenizer.token_to_id(TURN_END))
    return NativeConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=config.width,
        num_attention_heads=config.heads,
        num_key_value_heads=config.kv_heads or config.heads,
        num_hidden_layers=config.layers,
        intermediate_size=config.feed_forward_units(),
        max_position_embeddings=config.context,
        rope_theta=config.rope_theta,
        qk_norm=config.qk_norm,
        attention_gate=config.attention_gate,
        document_start=config.document_start,
        value_embeddings=config.value_embeddings,
        residual_scalars=config.residual_scalars,
        feed_forward=config.feed_forward,
        count_table=None
        if not config.count_table
        else {
            "orders": table.manifest["orders"],
            "candidates": table.manifest["candidates"],
            "rows": [len(table.tensors[f"keys_{length}"]) for length in range(1, table.manifest["orders"])],
            "document_start": table.manifest["document_start"],
            "candidates_dtype": str(table.tensors["candidates_0"].dtype).removeprefix("torch."),
            "digest": table.digest,
        },
        bos_token_id=tokenizer.token_to_id(BOS),
        eos_token_id=ends,
        pad_token_id=tokenizer.token_to_id(EOS),
    )


def export_huggingface(checkpoint, output):
    """Write the checkpoint's model, tokenizer and loading code to `output`."""
    from transformers import PreTrainedTokenizerFast

    from .hf.configuration_native import NativeConfig
    from .hf.modeling_native import NativeForCausalLM

    payload = load_checkpoint(checkpoint)
    tokenizer = Tokenizer.from_str(payload["tokenizer"])
    model_config = ModelConfig(**payload["model_config"])
    table = saved_count_table(payload) if model_config.count_table else None
    config = native_config(model_config, tokenizer, table)
    NativeConfig.register_for_auto_class()
    NativeForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    # vLLM's Transformers backend loads the layers through AutoModel.
    config.auto_map = {"AutoModel": "modeling_native.NativeModel"}
    model = NativeForCausalLM(config)
    # Multi-token prediction only trains; the logits come from the main layers.
    weights = {name: value for name, value in payload["model"].items() if not name.startswith("predictors.")}
    if table is not None:
        weights.update({f"count_{name}": tensor for name, tensor in table.tensors.items()})
    model.model.load_state_dict(weights)
    output = Path(output)
    model.save_pretrained(output)
    bos = tokenizer.token_to_id(BOS)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", pair=f"{BOS} $A $B:1", special_tokens=[(BOS, bos)]
    )
    markers = all(tokenizer.token_to_id(marker) is not None for marker in MARKERS)
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS,
        eos_token=EOS,
        pad_token=EOS,
        chat_template=CHAT_TEMPLATE if markers else None,
    ).save_pretrained(output)
    # Transformers 5 records the class as TokenizersBackend, which Transformers 4, as vLLM installs it, does not
    # know; both load the older name.
    settings = output / "tokenizer_config.json"
    settings.write_text(
        json.dumps({**json.loads(settings.read_text(encoding="utf-8")), "tokenizer_class": "PreTrainedTokenizerFast"},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"output": str(output), "step": payload["step"], "parameters": model.num_parameters()}
