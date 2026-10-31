import dataclasses
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.gemma4 import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig

from pretrain.conversation import chat_prompt
from pretrain.data import BOS
from pretrain.hf_serving import HuggingFaceModel
from pretrain.huggingface import export_huggingface
from pretrain.model import LanguageModel, ModelConfig
from pretrain.remote import GPUTime, RemoteTextModel, relay
from responses_api import ModelSettings
from server import create_app
from tests.test_benchmarks import byte_tokenizer

MARKERS = [
    "<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>",
    "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>',
]
# A short chat template in the form of Gemma 4's.
CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- if enable_thinking or tools -%}"
    "{{- '<|turn>system\\n' -}}"
    "{%- if enable_thinking -%}<|think|>{%- endif -%}"
    "{%- for tool in tools or [] -%}<|tool>{{- tool.function.name -}}<tool|>{%- endfor -%}"
    "{{- '<turn|>\\n' -}}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "{{- '<|turn>' + ('model' if message.role == 'assistant' else message.role) + '\\n' + message.content -}}"
    "{%- for call in message.tool_calls or [] -%}<|tool_call>call:{{- call.function.name -}}{}<tool_call|>{%- endfor -%}"
    "{%- if message.role == 'tool' -%}<|tool_response>{{- message.name -}}<tool_response|>{%- endif -%}"
    "{{- '<turn|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|turn>model\\n' -}}"
    "{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}"
    "{%- endif -%}"
)
# The response template of Gemma 4's tokenizer.
RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {"close": ["<turn|>", "<|tool_response>", "<eos>"], "content": "text"},
        "thinking": {"close": "<channel|>", "content": "text", "open": "<|channel>thought\n"},
        "tool_calls": {
            "close": "<tool_call|>",
            "content": "json",
            "content_args": {"string_delims": [['<|"|>', '<|"|>']], "unquoted_keys": True},
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "repeats": True,
            "transform": {"function": {"arguments": "{content}", "name": "{name}"}, "type": "function"},
        },
    },
    "start_anchor": ["<|turn>model\n", "<tool_response|>"],
}
USER = [{"role": "user", "content": "天気は？"}]
WEATHER = {"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
OFFERING = [{"role": "system", "content": "", "tools": [WEATHER]}, *USER]


def tiny_gemma(directory):
    """Save a tokenizer and a tiny Gemma 4 with random weights to `directory`."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["日本語 English thought weather city"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=MARKERS,
            show_progress=False,
        ),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        chat_template=CHAT_TEMPLATE,
        response_template=RESPONSE_TEMPLATE,
    ).save_pretrained(directory)
    size = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}
    config = Gemma4Config(
        text_config=Gemma4TextConfig(
            vocab_size=tokenizer.get_vocab_size(), num_key_value_heads=1, head_dim=8, **size,
        ),
        vision_config=Gemma4VisionConfig(**size),
        audio_config=None,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.generation_config.eos_token_id = [tokenizer.token_to_id("<eos>"), tokenizer.token_to_id("<turn|>")]
    model.generation_config.pad_token_id = tokenizer.token_to_id("<pad>")
    model.save_pretrained(directory)


class HuggingFaceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        tiny_gemma(directory.name)
        cls.backend = HuggingFaceModel(directory.name, context=256)

    def scripted(self, *pieces):
        """Make `generate` write these pieces, token by token, and return what it was asked."""
        tokenizer = self.backend.tokenizer
        tokens = [token for piece in pieces for token in tokenizer(piece, add_special_tokens=False)["input_ids"]]
        asked = {}

        def generate(**arguments):
            asked.update(arguments)
            streamer = arguments["streamer"]
            streamer.put(arguments["input_ids"])
            for token in tokens[: arguments["max_new_tokens"]]:
                streamer.put(torch.tensor([token]))
            streamer.end()

        patcher = patch.object(self.backend.model, "generate", generate)
        patcher.start()
        self.addCleanup(patcher.stop)
        return asked

    def reply(self, messages=USER, limit=32, temperature=1.0, top_p=1.0, **options):
        """Return the generation, closed, and the deltas it wrote."""
        generation = self.backend.start(messages, limit, temperature, top_p, **options)
        try:
            return generation, list(generation)
        finally:
            generation.close()

    def prompt(self, asked):
        return self.backend.tokenizer.decode(asked["input_ids"][0])

    def test_writes_the_prompt_with_the_tokenizers_chat_template(self):
        """Writes the prompt with the tokenizer's chat template, with one token at the start."""
        asked = self.scripted("雨です。")
        self.reply([{"role": "system", "content": "短く答える。"}, *USER])
        self.assertEqual(
            self.prompt(asked),
            "<bos><|turn>system\n短く答える。<turn|>\n<|turn>user\n天気は？<turn|>\n"
            "<|turn>model\n<|channel>thought\n<channel|>",
        )

    def test_asks_for_reasoning_through_the_chat_template(self):
        """Lets the reply open its reasoning when the request asks for reasoning."""
        asked = self.scripted("雨です。")
        self.reply(reasoning=True)
        self.assertEqual(
            self.prompt(asked), "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\n天気は？<turn|>\n<|turn>model\n"
        )

    def test_returns_the_text_and_counts_the_tokens_of_a_finished_reply(self):
        """Returns the text of a reply that ends itself, counting the token that ends it."""
        self.scripted("雨です。", "<turn|>", "続き")
        generation, deltas = self.reply()
        self.assertEqual("".join(text for _, text in deltas), "雨です。")
        self.assertEqual({phase for phase, _ in deltas}, {"text"})
        self.assertEqual(generation.text, "雨です。")
        self.assertTrue(generation.finished)
        tokenizer = self.backend.tokenizer
        self.assertEqual(generation.input_tokens, len(tokenizer.apply_chat_template(USER, add_generation_prompt=True, tokenize=True)["input_ids"]))
        self.assertEqual(generation.output_tokens, len(tokenizer("雨です。<turn|>", add_special_tokens=False)["input_ids"]))

    def test_keeps_characters_whole_across_tokens(self):
        """Returns only whole characters while their bytes arrive over several tokens."""
        self.scripted("😀日本語", "<turn|>")
        _, deltas = self.reply()
        self.assertEqual("".join(text for _, text in deltas), "😀日本語")
        self.assertNotIn("�", "".join(text for _, text in deltas))

    def test_leaves_a_reply_unfinished_at_its_limit(self):
        """Stops at the limit and leaves the reply unfinished."""
        self.scripted("日本語 English 日本語 English")
        generation, _ = self.reply(limit=3)
        self.assertFalse(generation.finished)
        self.assertEqual(generation.output_tokens, 3)

    def test_returns_an_empty_reply_that_ends_at_once(self):
        """Returns no deltas for a reply that ends at once."""
        self.scripted("<turn|>")
        generation, deltas = self.reply()
        self.assertEqual(deltas, [])
        self.assertTrue(generation.finished)

    def test_separates_the_reasoning_from_the_text(self):
        """Returns reasoning and text apart, counting the tokens of the reasoning."""
        self.scripted("<|channel>thought\n考える。", "<channel|>", "雨です。", "<turn|>")
        generation, deltas = self.reply(reasoning=True)
        self.assertEqual("".join(text for phase, text in deltas if phase == "reasoning"), "考える。")
        self.assertEqual("".join(text for phase, text in deltas if phase == "text"), "雨です。")
        self.assertEqual(generation.reasoning_text, "考える。")
        self.assertEqual(generation.text, "雨です。")
        self.assertGreater(generation.reasoning_tokens, 0)

    def test_answers_directly_when_the_reply_does_not_reason(self):
        """Returns everything as text when a reply asked to reason answers directly."""
        self.scripted("雨です。", "<turn|>")
        generation, deltas = self.reply(reasoning=True)
        self.assertEqual({phase for phase, _ in deltas}, {"text"})
        self.assertEqual(generation.reasoning_tokens, 0)

    def test_leaves_out_reasoning_the_request_did_not_ask_for(self):
        """Leaves out the reasoning of a reply when the request did not ask for it."""
        self.scripted("<|channel>thought\n考える。", "<channel|>", "雨です。", "<turn|>")
        generation, deltas = self.reply()
        self.assertEqual({phase for phase, _ in deltas}, {"text"})
        self.assertEqual(generation.text, "雨です。")
        self.assertEqual(generation.reasoning_text, "")

    def test_lists_the_tools_and_earlier_calls_in_the_prompt(self):
        """Gives the template the offered tools, earlier calls and tool outputs in its form."""
        asked = self.scripted("雨です。")
        self.reply(
            [
                *OFFERING,
                {"role": "assistant", "content": "", "tool_calls": [{"name": "weather", "arguments": {"city": "東京"}}]},
                {"role": "tool", "content": "雨", "name": "weather"},
            ],
            tools=["weather"],
        )
        prompt = self.prompt(asked)
        self.assertIn("<|tool>weather<tool|>", prompt)
        self.assertIn("<|tool_call>call:weather{}<tool_call|>", prompt)
        self.assertIn("<|tool_response>weather<tool_response|>", prompt)

    def test_returns_calls_of_the_offered_tools(self):
        """Returns a call of an offered tool with its name and arguments, between the reply's text."""
        self.scripted("見ます。", '<|tool_call>call:weather{city:<|"|>東京<|"|>}<tool_call|>', "<turn|>")
        generation, deltas = self.reply(OFFERING, limit=64, tools=["weather"])
        self.assertEqual(generation.text, "見ます。")
        self.assertEqual(deltas[-1], ("call", {"name": "weather", "arguments": {"city": "東京"}}))
        self.assertTrue(generation.finished)

    def test_keeps_a_call_of_anything_else_in_the_text(self):
        """Keeps a call of a tool that was not offered in the reply's text."""
        self.scripted('<|tool_call>call:clock{zone:<|"|>JST<|"|>}<tool_call|>', "<turn|>")
        generation, deltas = self.reply(OFFERING, tools=["weather"])
        self.assertEqual({phase for phase, _ in deltas}, {"text"})
        self.assertEqual(json.loads(generation.text), {"name": "clock", "arguments": {"zone": "JST"}})

    def test_keeps_a_call_cut_off_at_the_limit_in_the_text(self):
        """Keeps a call that the limit cuts off in the reply's text."""
        self.scripted('<|tool_call>call:weather{city:<|"|>東京')
        generation, deltas = self.reply(OFFERING, tools=["weather"])
        self.assertEqual({phase for phase, _ in deltas}, {"text"})
        self.assertIn("東京", generation.text)
        self.assertFalse(generation.finished)

    def test_goes_on_with_the_reply_after_a_call_that_does_not_parse(self):
        """Keeps a call that does not parse in the reply's text, and goes on with the reply."""
        self.scripted('<|tool_call>call:weather{city:<|"|>東京<tool_call|>', "続きです。", "<turn|>")
        generation, deltas = self.reply(OFFERING, limit=64, tools=["weather"])
        self.assertEqual({phase for phase, _ in deltas}, {"text"})
        self.assertIn("東京", generation.text)
        self.assertTrue(generation.text.endswith("続きです。"))
        self.assertTrue(generation.finished)

    def test_ends_the_reply_at_its_first_call_when_asked(self):
        """Ends the reply at its first call when the request allows only one, and calls it finished."""
        self.scripted(
            '<|tool_call>call:weather{city:<|"|>東京<|"|>}<tool_call|>',
            '<|tool_call>call:weather{city:<|"|>大阪<|"|>}<tool_call|>',
        )
        generation, deltas = self.reply(OFFERING, limit=64, tools=["weather"], single_call=True)
        self.assertEqual(deltas, [("call", {"name": "weather", "arguments": {"city": "東京"}})])
        self.assertTrue(generation.finished)

    def test_refuses_an_input_beyond_the_context_before_generating(self):
        """Refuses an input that leaves no room for the reply, before it starts generating."""
        with patch.object(self.backend.model, "generate") as generate, self.assertRaisesRegex(ValueError, "context length 256"):
            self.backend.start(USER, 256, 1.0, 1.0)
        generate.assert_not_called()

    def test_asks_generate_to_sample_at_a_temperature_and_otherwise_takes_the_likeliest_token(self):
        """Samples above a temperature of zero, and takes the likeliest token at zero."""
        asked = self.scripted("雨")
        self.reply(temperature=0.5, top_p=0.9)
        self.assertEqual((asked["do_sample"], asked["temperature"], asked["top_p"]), (True, 0.5, 0.9))
        self.reply(temperature=0)
        self.assertFalse(asked["do_sample"])

    def test_reports_a_failure_of_generate_to_the_reader(self):
        """Raises to the reader what `generate` failed with."""
        with patch.object(self.backend.model, "generate", side_effect=RuntimeError("out of memory")):
            generation = self.backend.start(USER, 8, 1.0, 1.0)
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                list(generation)
            generation.close()

    def test_writes_with_the_real_model_as_generate_does(self):
        """Writes the tokens the model's own `generate` writes, at a temperature of zero."""
        generation = self.backend.start(USER, 8, 0, 1.0)
        list(generation)
        generation.close()
        tokenizer = self.backend.tokenizer
        tokens = tokenizer.apply_chat_template(USER, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        expected = self.backend.model.generate(**tokens, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None)
        self.assertEqual(generation.output_tokens, expected.shape[1] - tokens["input_ids"].shape[1])

    def test_closing_stops_generating(self):
        """Stops generating when it is closed in the middle of a reply."""
        generation = self.backend.start(USER, 200, 1.0, 1.0)
        generation.close()
        self.assertFalse(generation.thread.is_alive())
        self.assertLess(generation.written.tokens.qsize(), 200)

    def serve(self):
        """A client of an app whose model samples elsewhere, its records serialized as between containers."""

        def sample(*arguments):
            for record in relay(self.backend, *arguments):
                yield pickle.loads(pickle.dumps(record))

        backend = RemoteTextModel(sample, "f" * 40, GPUTime({}, 3600, 60))
        settings = ModelSettings(model_id="test-model", max_output_tokens=32, top_p=0.95)
        client = TestClient(create_app(response_settings=settings, backend=backend))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_serves_the_reply_and_its_reasoning_through_the_api(self):
        """Serves a reply and its reasoning through the API, as JSON and as a stream."""
        self.scripted("<|channel>thought\n考える。", "<channel|>", "雨です。", "<turn|>")
        client = self.serve()
        request = {"model": "test-model", "input": "天気は？", "max_output_tokens": 32, "reasoning": {"effort": "low"}}
        response = client.post("/v1/responses", json=request).json()
        self.assertEqual(response["status"], "completed")
        self.assertEqual([item["type"] for item in response["output"]], ["reasoning", "message"])
        self.assertEqual(response["output"][0]["content"][0]["text"], "考える。")
        self.assertEqual(response["output"][1]["content"][0]["text"], "雨です。")
        self.assertEqual(response["top_p"], 0.95)
        with client.stream("POST", "/v1/responses", json={**request, "stream": True}) as stream:
            events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(
            "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta"), "雨です。"
        )

    def test_serves_a_call_through_the_api(self):
        """Serves a call of an offered tool through the API."""
        self.scripted('<|tool_call>call:weather{city:<|"|>東京<|"|>}<tool_call|>', "<turn|>")
        response = self.serve().post(
            "/v1/responses",
            json={"model": "test-model", "input": "天気は？", "tools": [{"type": "function", **WEATHER}]},
        ).json()
        self.assertEqual(response["output"][-1]["type"], "function_call")
        self.assertEqual(response["output"][-1]["name"], "weather")
        self.assertEqual(json.loads(response["output"][-1]["arguments"]), {"city": "東京"})


class ExportedModelTests(unittest.TestCase):
    def test_serves_a_model_exported_from_our_checkpoints(self):
        """Serves an exported checkpoint in its own conversation format, with BOS first and the reply as text."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        tokenizer = byte_tokenizer(reserved_tokens=8)
        config = ModelConfig(
            architecture="llama", width=32, heads=4, kv_heads=2, layers=2, context=256,
            qk_norm=True, attention_gate="headwise", document_start=tokenizer.token_to_id(BOS), prediction_depth=1,
        )
        native = LanguageModel(tokenizer.get_vocab_size(), config)
        torch.save(
            {"version": 1, "model": native.state_dict(), "model_config": dataclasses.asdict(config),
             "tokenizer": tokenizer.to_str(), "step": 3, "tokens_seen": 10},
            root / "last.pt",
        )
        export_huggingface(root / "last.pt", root / "hf")
        backend = HuggingFaceModel(root / "hf", trust_remote_code=True)
        generation = backend.start(USER, 8, 1.0, 1.0)
        deltas = list(generation)
        generation.close()
        self.assertEqual({phase for phase, _ in deltas} - {"text"}, set())
        self.assertEqual(
            generation.input_tokens, 1 + len(tokenizer.encode(chat_prompt(USER), add_special_tokens=False).ids)
        )


if __name__ == "__main__":
    unittest.main()
