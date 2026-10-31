import itertools
import json
import pickle
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import openai
import torch
import uvicorn
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.conversation import CALL_END, CALL_START, THINK_END, TURN_END, chat_prompt
from pretrain.model import LanguageModel, ModelConfig
from pretrain.remote import GPUTime, OutOfGPUTime, RemoteTextModel, relay
from pretrain.serving import TextModel
from responses_api import ModelSettings
from server import create_app

WEATHER = {
    "type": "function",
    "name": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}
NOW = {"type": "function", "name": "now"}
WEATHER_CALL = '{"name": "weather", "arguments": {"city": "東京"}}'
# Noon UTC, half a day before the next day starts.
NOON = 20000 * 86400 + 43200


class ResponsesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name) / "model.pt"
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.train_from_iterator(
            ["日本語 English"],
            trainers.BpeTrainer(
                vocab_size=266,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                # The special tokens, then the reserved tokens that mark conversations.
                special_tokens=["<|bos|>", "<|eos|>", *(f"<|reserved_{index}|>" for index in range(8))],
                show_progress=False,
            ),
        )
        config = ModelConfig(
            width=8, heads=2, layers=1, context=256, architecture="llama"
        )
        model = LanguageModel(self.tokenizer.get_vocab_size(), config)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 2:
                    parameter.zero_()
            model.embedding.weight[:, 0] = 1
            model.embedding.weight[self.tokenizer.token_to_id("a"), 0] = 2
        torch.save(
            {
                "version": 1,
                "model": model.state_dict(),
                "model_config": asdict(config),
                "tokenizer": self.tokenizer.to_str(),
                "step": 7,
                "tokens_seen": 1024,
            },
            self.checkpoint,
        )
        self.settings = ModelSettings(
            model_id="test-model", checkpoint=self.checkpoint, max_output_tokens=96
        )
        self.app = create_app(response_settings=self.settings)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def post(self, **values):
        return self.client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "日本語 English",
                "max_output_tokens": 4,
                "temperature": 0,
                **values,
            },
        )

    def sdk(self):
        listening = socket.socket()
        listening.bind(("127.0.0.1", 0))
        listening.listen()
        config = uvicorn.Config(
            create_app(response_settings=self.settings),
            log_level="critical",
            access_log=False,
            lifespan="on",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listening]}, daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                self.fail("HTTP server did not start")
            time.sleep(0.01)

        def stop():
            server.should_exit = True
            thread.join(timeout=5)
            listening.close()

        self.addCleanup(stop)
        client = openai.OpenAI(
            base_url="http://127.0.0.1:" + str(listening.getsockname()[1]) + "/v1",
            api_key="local-test-key",
            max_retries=0,
            timeout=5,
            _strict_response_validation=True,
        )
        self.addCleanup(client.close)
        return client, config.app

    def test_generates_a_response_with_native_weights_and_actual_token_usage(self):
        """Generates with native weights and returns the status, revision and token usage."""
        result = self.post(metadata={"purpose": "test"})
        self.assertEqual(result.status_code, 200, result.text)
        response = result.json()
        self.assertEqual(response["object"], "response")
        self.assertTrue(response["id"].startswith("resp_"))
        self.assertEqual(response["model"], "test-model")
        self.assertEqual(len(response["model_revision"]), 64)
        self.assertEqual(response["output"][0]["content"][0]["text"], "aaaa")
        self.assertEqual(response["output"][0]["role"], "assistant")
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(
            response["incomplete_details"], {"reason": "max_output_tokens"}
        )
        self.assertEqual(response["metadata"], {"purpose": "test"})
        self.assertEqual(response["usage"]["output_tokens"], 4)
        expected = 1 + len(
            self.tokenizer.encode(chat_prompt([{"role": "user", "content": "日本語 English"}])).ids
        )
        self.assertEqual(response["usage"]["input_tokens"], expected)
        self.assertEqual(response["usage"]["total_tokens"], expected + 4)

    def test_accepts_text_messages_instructions_and_previous_assistant_output(self):
        """Accepts text history and instructions and reuses assistant output as input."""
        first = self.post(
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": "日本語"}]}
            ]
        )
        second = self.post(
            instructions="Continue in Japanese.",
            input=[
                {"role": "user", "content": "日本語"},
                *first.json()["output"],
                {"role": "user", "content": "English"},
            ],
            text={"format": {"type": "text"}},
        )
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["instructions"], "Continue in Japanese.")
        self.assertEqual(second.json()["output"][0]["content"][0]["text"], "aaaa")
        prompt = chat_prompt(
            [
                {"role": "system", "content": "Continue in Japanese."},
                {"role": "user", "content": "日本語"},
                {"role": "assistant", "content": "aaaa"},
                {"role": "user", "content": "English"},
            ]
        )
        self.assertEqual(
            second.json()["usage"]["input_tokens"],
            1 + len(self.tokenizer.encode(prompt).ids),
        )

    def test_reports_completion_when_the_model_emits_eos(self):
        """Returns completion and token usage when the model emits EOS."""
        self.post()
        model = self.app.state.responses.backend.model
        eos = self.tokenizer.token_to_id("<|eos|>")
        with torch.no_grad():
            model.embedding.weight[eos, 0] = 3
        response = self.post().json()
        self.assertEqual(response["status"], "completed")
        self.assertIsInstance(response["completed_at"], int)
        self.assertIsNone(response["incomplete_details"])
        self.assertEqual(response["usage"]["output_tokens"], 1)

    def remote(self, gpu_time=None):
        """A client of an app whose model samples elsewhere, its records serialized as between containers."""
        model = TextModel(self.checkpoint)

        def sample(*arguments):
            for record in relay(model, *arguments):
                yield pickle.loads(pickle.dumps(record))

        backend = RemoteTextModel(sample, "f" * 40, gpu_time or GPUTime({}, 3600, 60))
        client = TestClient(create_app(response_settings=self.settings, backend=backend))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client, model

    def scripted(self, *pieces, app=None, model=None):
        """Make the model emit these texts and reserved tokens, one token per step."""
        tokens = iter(
            [
                token
                for piece in pieces
                for token in (
                    [self.tokenizer.token_to_id(piece)]
                    if self.tokenizer.token_to_id(piece) is not None
                    else self.tokenizer.encode(piece).ids
                )
            ]
        )

        def scripted_logits(window, start, caches):
            logits = torch.full((self.tokenizer.get_vocab_size(),), -torch.inf)
            logits[next(tokens)] = 1
            return logits

        return patch.object(
            model or (app or self.app).state.responses.backend.model, "cached_logits", side_effect=scripted_logits
        )

    def test_returns_reasoning_before_the_answer_when_asked_to_reason(self):
        """推論を頼まれると考える過程から始め、考えた内容を回答とは別の項目で返し、そのトークン数を数える。"""
        self.post()
        with self.scripted("2と2", THINK_END, "4", TURN_END):
            response = self.post(input="2+2?", reasoning={"effort": "medium"}, max_output_tokens=16).json()
        self.assertEqual(response["status"], "completed")
        thought, message = response["output"]
        self.assertEqual(thought["type"], "reasoning")
        self.assertEqual(thought["content"], [{"type": "reasoning_text", "text": "2と2"}])
        self.assertEqual(message["content"][0]["text"], "4")
        self.assertEqual(
            response["usage"]["output_tokens_details"]["reasoning_tokens"],
            len(self.tokenizer.encode("2と2").ids) + 1,
        )
        self.assertEqual(response["reasoning"]["effort"], "medium")
        self.assertEqual(
            response["usage"]["input_tokens"],
            1 + len(self.tokenizer.encode(chat_prompt([{"role": "user", "content": "2+2?"}], reasoning=True)).ids),
        )

    def test_streams_reasoning_before_the_answer_and_drops_earlier_reasoning(self):
        """推論の内容を回答より先に流し、前の応答の推論の項目を入力に含めても会話には入れない。"""
        self.post()
        earlier = {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "x"}]}
        with self.scripted("2と2", THINK_END, "4", TURN_END):
            response = self.post(
                input=[{"role": "user", "content": "1+1?"}, earlier, {"role": "assistant", "content": "2"},
                       {"role": "user", "content": "2+2?"}],
                reasoning={"effort": "high"},
                max_output_tokens=16,
                stream=True,
            )
        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        kinds = [event["type"] for event in events]
        self.assertLess(kinds.index("response.reasoning_text.delta"), kinds.index("response.reasoning_text.done"))
        self.assertLess(kinds.index("response.reasoning_text.done"), kinds.index("response.output_text.delta"))
        final = events[-1]["response"]
        self.assertEqual([item["type"] for item in final["output"]], ["reasoning", "message"])
        self.assertEqual(final["output"][1]["content"][0]["text"], "4")
        history = [
            {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}, {"role": "user", "content": "2+2?"}
        ]
        self.assertEqual(
            final["usage"]["input_tokens"],
            1 + len(self.tokenizer.encode(chat_prompt(history, reasoning=True)).ids),
        )

    def test_lists_tools_in_the_system_turn_and_returns_calls_as_items(self):
        """渡した道具をシステムの発言に並べ、モデルの呼び出しを、名前と引数の JSON 文字列を持つ項目で返す。"""
        self.post()
        with self.scripted(CALL_START, WEATHER_CALL, CALL_END, TURN_END):
            response = self.post(input="天気は？", tools=[WEATHER], max_output_tokens=64).json()
        self.assertEqual(response["status"], "completed")
        [call] = response["output"]
        self.assertEqual((call["type"], call["name"], call["status"]), ("function_call", "weather", "completed"))
        self.assertEqual(json.loads(call["arguments"]), {"city": "東京"})
        self.assertTrue(call["call_id"].startswith("call_"))
        self.assertEqual(response["tools"][0]["name"], "weather")
        tools = [{"name": "weather", "parameters": WEATHER["parameters"]}]
        prompt = chat_prompt([{"role": "system", "content": "", "tools": tools}, {"role": "user", "content": "天気は？"}])
        self.assertEqual(response["usage"]["input_tokens"], 1 + len(self.tokenizer.encode(prompt).ids))

    def test_returns_calls_of_tools_not_offered_and_unreadable_calls_as_text(self):
        """渡していない道具の呼び出しと、JSON として読めない呼び出しは、本文として返す。"""
        self.post()
        for written in ('{"name": "search", "arguments": {}}', '{"name": "weather"'):
            with self.subTest(written=written), self.scripted(CALL_START, written, CALL_END, TURN_END):
                response = self.post(input="天気は？", tools=[WEATHER], max_output_tokens=64).json()
                self.assertEqual([item["type"] for item in response["output"]], ["message"])
                self.assertEqual(response["output"][0]["content"][0]["text"], written)

    def test_ends_the_reply_at_its_first_call_when_parallel_calls_are_off(self):
        """parallel_tool_calls が false のときは、最初の呼び出しで応答を終える。"""
        self.post()
        call = '{"name": "now", "arguments": {}}'
        pieces = (CALL_START, call, CALL_END, CALL_START, call, CALL_END, TURN_END)
        for parallel, calls in ((True, 2), (False, 1)):
            with self.subTest(parallel=parallel), self.scripted(*pieces):
                response = self.post(input="今は？", tools=[NOW], parallel_tool_calls=parallel, max_output_tokens=96).json()
                self.assertEqual(response["status"], "completed")
                self.assertEqual([item["type"] for item in response["output"]], ["function_call"] * calls)

    def test_opens_no_call_without_tools_it_may_call(self):
        """道具を渡さないときと tool_choice が none のときは、モデルが呼び出しを始めようとしても文章を書かせる。"""
        self.post()
        model = self.app.state.responses.backend.model
        with torch.no_grad():
            model.embedding.weight[self.tokenizer.token_to_id(CALL_START), 0] = 3
        for fields, text in (({}, "aaaa"), ({"tools": [NOW], "tool_choice": "none"}, "aaaa"), ({"tools": [NOW]}, "")):
            with self.subTest(fields=fields):
                response = self.post(**fields).json()
                self.assertEqual(response["output"][0]["content"][0]["text"], text)

    def test_streams_calls_and_continues_from_their_outputs_through_the_official_sdk(self):
        """公式 SDK で、文章のあとの呼び出しを流して受け取り、その結果を渡すと、呼び出しと結果を会話に入れて答える。"""
        sdk, app = self.sdk()
        sdk.responses.create(model="test-model", input="hello", max_output_tokens=1)
        with (
            self.scripted("見ます", CALL_START, WEATHER_CALL, CALL_END, TURN_END, app=app),
            sdk.responses.stream(model="test-model", input="天気は？", tools=[WEATHER], max_output_tokens=96) as stream,
        ):
            kinds = [event.type for event in stream]
            first = stream.get_final_response()
        self.assertLess(kinds.index("response.output_text.done"), kinds.index("response.function_call_arguments.delta"))
        message, call = first.output
        self.assertEqual(message.content[0].text, "見ます")
        self.assertEqual((call.type, call.name, json.loads(call.arguments)), ("function_call", "weather", {"city": "東京"}))
        result = {"type": "function_call_output", "call_id": call.call_id, "output": "晴れ"}
        with self.scripted("晴れ", TURN_END, app=app):
            second = sdk.responses.create(
                model="test-model",
                input=[{"role": "user", "content": "天気は？"}, *first.output, result],
                tools=[WEATHER],
                max_output_tokens=8,
            )
        self.assertEqual(second.output_text, "晴れ")
        prompt = chat_prompt(
            [
                {"role": "system", "content": "", "tools": [{"name": "weather", "parameters": WEATHER["parameters"]}]},
                {"role": "user", "content": "天気は？"},
                {"role": "assistant", "content": "見ます", "tool_calls": [{"name": "weather", "arguments": {"city": "東京"}}]},
                {"role": "tool", "content": "晴れ"},
            ]
        )
        self.assertEqual(second.usage.input_tokens, 1 + len(self.tokenizer.encode(prompt).ids))

    def test_streams_complete_japanese_characters_and_final_usage(self):
        """Streams complete Japanese characters and returns the final text and usage."""
        self.post()
        model = self.app.state.responses.backend.model
        tokens = iter(
            [
                *self.tokenizer.encode("日本語").ids,
                self.tokenizer.token_to_id("<|eos|>"),
            ]
        )

        def scripted_logits(window, start, caches):
            logits = torch.full((self.tokenizer.get_vocab_size(),), -torch.inf)
            logits[next(tokens)] = 1
            return logits

        with patch.object(model, "cached_logits", side_effect=scripted_logits):
            response = self.post(stream=True, max_output_tokens=16)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["Content-Type"].startswith("text/event-stream")
        )
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )
        text = "".join(
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        self.assertEqual(text, "日本語")
        self.assertEqual(events[-1]["type"], "response.completed")
        final = events[-1]["response"]
        self.assertEqual(final["output"][0]["content"][0]["text"], text)
        self.assertEqual(
            final["usage"]["output_tokens"], len(self.tokenizer.encode(text).ids) + 1
        )

    def test_validates_options_and_returns_openai_errors(self):
        """Reports invalid and unsupported parameters and accepts subsequent valid requests."""
        for fields in (
            {"temperature": -1},
            {"temperature": "1"},
            {"top_p": 0},
            {"max_output_tokens": True},
            {"max_output_tokens": 97},
            {"unknown": True},
            {"store": True},
            {"previous_response_id": "resp_old"},
            {"background": True},
            {"tools": [{"type": "web_search"}]},
            {"tools": [NOW, NOW]},
            {"tool_choice": "required"},
            {
                "input": [
                    {"role": "user", "content": "x"},
                    {"type": "function_call", "call_id": "call_1", "name": "now", "arguments": "[]"},
                ]
            },
            {
                "input": [
                    {"role": "user", "content": "x"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "y"},
                ]
            },
            {"text": {"format": {"type": "json_object"}}},
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/image",
                            }
                        ],
                    }
                ]
            },
            {"metadata": {"number": 42}},
        ):
            with self.subTest(fields=fields):
                response = self.post(**fields)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(
                    response.json()["error"]["type"], "invalid_request_error"
                )
                self.assertTrue(response.json()["error"]["code"])
        self.assertEqual(self.post().status_code, 200)

    def test_handles_malformed_json_content_types_and_body_limits(self):
        """Returns OpenAI errors for malformed JSON, invalid numbers, types and body sizes."""
        for body in (
            "{",
            '{"model":"test-model","input":"hello","temperature":NaN}',
            '{"model":"test-model","input":"hello","top_p":1e309}',
        ):
            response = self.client.post(
                "/v1/responses",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertTrue(response.json()["error"]["message"])
        response = self.client.post("/v1/responses", content="{}")
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["error"]["code"], "invalid_content_type")
        response = self.client.post(
            "/v1/responses",
            content=b" " * 65537,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "request_too_large")

    def test_streams_replies_sampled_elsewhere_as_the_local_model_does(self):
        """別の場所で生成した返事も、考えた内容・文章・道具の呼び出しを手元と同じ順で流し、同じ応答とトークン数を返す。"""
        self.post()
        client, model = self.remote()
        pieces = ("2と2", THINK_END, "見ます", CALL_START, WEATHER_CALL, CALL_END, TURN_END)
        request = {
            "model": "test-model",
            "input": "天気は？",
            "tools": [WEATHER],
            "reasoning": {"effort": "medium"},
            "max_output_tokens": 96,
            "temperature": 0,
            "stream": True,
        }
        streams = []
        for target, sampler in ((self.client, self.app.state.responses.backend.model), (client, model.model)):
            with self.scripted(*pieces, model=sampler):
                response = target.post("/v1/responses", json=request)
            self.assertEqual(response.status_code, 200, response.text)
            streams.append([json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")])

        def outcome(events):
            final = events[-1]["response"]
            items = [{key: value for key, value in item.items() if key not in ("id", "call_id")} for item in final["output"]]
            return [event["type"] for event in events], final["status"], items, final["usage"]

        local, elsewhere = streams
        self.assertEqual(outcome(elsewhere), outcome(local))
        self.assertEqual([item["type"] for item in outcome(elsewhere)[2]], ["reasoning", "message", "function_call"])
        self.assertEqual(elsewhere[-1]["response"]["model_revision"], "f" * 40)

    def test_refuses_inputs_beyond_the_context_before_sampling_elsewhere(self):
        """別の場所で生成するときも、文脈の長さを超える入力は生成を始める前に断る。"""
        client, _ = self.remote()
        response = client.post(
            "/v1/responses", json={"model": "test-model", "input": "a" * 256, "max_output_tokens": 4}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "context_length_exceeded")

    def test_counts_the_gpu_time_of_replies_and_of_inputs_the_gpu_refuses(self):
        """Counts from the request until the GPU shuts down, for replies and for inputs the GPU refuses."""
        reply = {"input": "日本語", "max_output_tokens": 4}
        for request, status in (
            (reply, 200),
            ({**reply, "stream": True}, 200),
            ({**reply, "input": "a" * 256}, 400),
        ):
            with self.subTest(request):
                store = {}
                # The clock moves ten seconds between the start and the end.
                client, _ = self.remote(GPUTime(store, 3600, 60, itertools.count(NOON, 10).__next__))
                response = client.post("/v1/responses", json={"model": "test-model", **request})
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(store[GPUTime.day(NOON)][0], 70)

    def test_does_not_count_requests_refused_before_they_reach_the_gpu(self):
        """Leaves out requests refused before they reach the GPU."""
        store = {}
        client, _ = self.remote(GPUTime(store, 3600, 60))
        response = client.post("/v1/responses", json={"model": "unknown", "input": "日本語"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(store, {})

    def test_refuses_replies_without_asking_the_gpu_once_its_time_has_run_out(self):
        """Keeps refusing replies without asking the GPU once its time has run out, giving the seconds until the next day."""
        store = {GPUTime.day(NOON): (3600.0, 0.0)}
        sample = Mock()
        backend = RemoteTextModel(sample, "f" * 40, GPUTime(store, 3600, 60, lambda: NOON))
        with TestClient(create_app(response_settings=self.settings, backend=backend)) as client:
            for _ in range(2):
                response = client.post("/v1/responses", json={"model": "test-model", "input": "日本語"})
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response.json()["error"]["code"], "insufficient_quota")
                self.assertEqual(response.headers["Retry-After"], "43200")
        sample.assert_not_called()
        self.assertEqual(store, {GPUTime.day(NOON): (3600.0, 0.0)})

    def test_reports_unknown_models_and_context_limits(self):
        """Reports unavailable models and exceeded context limits."""
        response = self.post(model="unknown")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")
        response = self.post(input="a" * 256)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "context_length_exceeded")
        self.assertEqual(self.post().status_code, 200)

    def test_keeps_health_available_during_generation_and_returns_retry_information(
        self,
    ):
        """Serves health during generation and provides retry information when busy."""
        self.post()
        model = self.app.state.responses.backend.model
        step = model.cached_logits
        started, release = threading.Event(), threading.Event()

        def delayed(*arguments):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Generation was not released")
            return step(*arguments)

        with (
            patch.object(model, "cached_logits", side_effect=delayed),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            pending = pool.submit(self.post)
            try:
                self.assertTrue(started.wait(timeout=5))
                response = self.post()
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response.headers["Retry-After"], "1")
                self.assertEqual(self.client.get("/healthz").status_code, 200)
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5).status_code, 200)
        self.assertEqual(self.post().status_code, 200)

    def test_authenticates_model_and_response_requests_with_the_configured_key(self):
        """Authenticates model listing and generation using the environment access key."""
        with patch.dict(
            "os.environ",
            {
                "MODEL_ID": "test-model",
                "MODEL_CHECKPOINT": str(self.checkpoint),
                "LANGUAGE_MODEL_API_KEY": "test-secret",
            },
            clear=True,
        ):
            settings = ModelSettings.from_env()
        with TestClient(create_app(response_settings=settings)) as client:
            self.assertEqual(client.get("/v1/models").status_code, 401)
            self.assertEqual(
                client.get(
                    "/v1/models", headers={"Authorization": "Bearer invalid"}
                ).status_code,
                401,
            )
            headers = {"Authorization": "Bearer test-secret"}
            self.assertEqual(client.get("/v1/models", headers=headers).status_code, 200)
            payload = {
                "model": "test-model",
                "input": "hello",
                "temperature": 0,
                "max_output_tokens": 1,
            }
            self.assertEqual(
                client.post("/v1/responses", json=payload).status_code, 401
            )
            self.assertEqual(
                client.post("/v1/responses", json=payload, headers=headers).status_code,
                200,
            )

    def test_documents_bearer_authentication_for_browser_api_requests(self):
        """Publishes Bearer authentication and a usable browser generation example."""
        settings = ModelSettings(
            model_id="test-model", checkpoint=self.checkpoint, api_key="test-secret"
        )
        with TestClient(create_app(response_settings=settings)) as client:
            specification = client.get("/openapi.json").json()
            for path, method in (("/v1/models", "get"), ("/v1/responses", "post")):
                operation = specification["paths"][path][method]
                scheme_name = next(iter(operation["security"][0]))
                scheme = specification["components"]["securitySchemes"][scheme_name]
                self.assertEqual(scheme["type"], "http")
                self.assertEqual(scheme["scheme"], "bearer")
                self.assertIn("言語モデルAPIのアクセスキー", scheme["description"])
            example = specification["paths"]["/v1/responses"]["post"]["requestBody"][
                "content"
            ]["application/json"]["example"]
            response = client.post(
                "/v1/responses",
                json=example,
                headers={"Authorization": "Bearer test-secret"},
            )
            self.assertEqual(response.status_code, 200, response.text)

    def test_loads_a_pinned_hub_checkpoint_and_serves_its_revision(self):
        """Restores a pinned HF checkpoint and returns its revision and generated text."""
        settings = ModelSettings(
            model_id="test-model",
            hub_repo="test-owner/test-model",
            hub_revision="f" * 40,
        )
        with (
            patch(
                "pretrain.hub.download_checkpoint",
                return_value={"checkpoint": str(self.checkpoint), "revision": "f" * 40},
            ) as download,
            TestClient(create_app(response_settings=settings)) as client,
        ):
            response = client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "hello",
                    "max_output_tokens": 2,
                    "temperature": 0,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model_revision"], "f" * 40)
        self.assertEqual(response.json()["output"][0]["content"][0]["text"], "aa")
        self.assertEqual(download.call_args.kwargs["revision"], "f" * 40)

    def test_reports_model_loading_failures_while_serving_health(self):
        """Returns 503 when loading fails and continues to serve health requests."""
        settings = ModelSettings(
            model_id="test-model", checkpoint=self.checkpoint.parent / "missing.pt"
        )
        with TestClient(create_app(response_settings=settings)) as client:
            for _ in range(2):
                response = client.post(
                    "/v1/responses", json={"model": "test-model", "input": "hello"}
                )
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"]["code"], "model_unavailable")
            self.assertEqual(client.get("/healthz").status_code, 200)

    def test_supports_openai_sdk_model_listing_and_response_output_text(self):
        """Lists models and retrieves generated text and usage through the official SDK."""
        sdk, _ = self.sdk()
        self.assertEqual(sdk.models.list().data[0].id, "test-model")
        prompt = chat_prompt([{"role": "system", "content": "Continue in Japanese."}, {"role": "user", "content": "日本語 English"}])
        cases = (
            {"input": "日本語 English", "instructions": "Continue in Japanese."},
            {
                "input": [
                    {"role": "system", "content": "Continue in Japanese."},
                    {"role": "user", "content": "日本語 English"},
                ]
            },
            {
                "input": [
                    {"role": "developer", "content": "Continue in Japanese."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "日本語"},
                            {"type": "input_text", "text": " English"},
                        ],
                    },
                ]
            },
        )
        for fields in cases:
            with self.subTest(fields=fields):
                response = sdk.responses.create(
                    model="test-model",
                    max_output_tokens=4,
                    temperature=0,
                    store=False,
                    **fields,
                )
                self.assertEqual(response.output_text, "aaaa")
                self.assertEqual(response.usage.output_tokens, 4)
                self.assertEqual(
                    response.usage.input_tokens,
                    1 + len(self.tokenizer.encode(prompt).ids),
                )
                self.assertEqual(response.status, "incomplete")
        with self.assertRaises(openai.BadRequestError):
            sdk.responses.create(model="test-model", input="hello", store=True)

    def test_supports_openai_sdk_stream_events_and_final_response(self):
        """Streams text and retrieves a completed response through the official SDK."""
        sdk, app = self.sdk()
        sdk.responses.create(model="test-model", input="hello", max_output_tokens=1)
        tokens = iter(
            [*self.tokenizer.encode("aaaa").ids, self.tokenizer.token_to_id("<|eos|>")]
        )

        def scripted_logits(window, start, caches):
            logits = torch.full((self.tokenizer.get_vocab_size(),), -torch.inf)
            logits[next(tokens)] = 1
            return logits

        with (
            patch.object(
                app.state.responses.backend.model,
                "cached_logits",
                side_effect=scripted_logits,
            ),
            sdk.responses.stream(
                model="test-model",
                input="日本語 English",
                instructions="Continue in Japanese.",
                max_output_tokens=8,
                temperature=0,
                store=False,
            ) as stream,
        ):
            text = "".join(
                event.delta
                for event in stream
                if event.type == "response.output_text.delta"
            )
            final = stream.get_final_response()
        self.assertEqual(text, "aaaa")
        self.assertEqual(final.output_text, text)
        self.assertEqual(final.status, "completed")
        self.assertEqual(final.usage.output_tokens, 5)
        prompt = chat_prompt([{"role": "system", "content": "Continue in Japanese."}, {"role": "user", "content": "日本語 English"}])
        self.assertEqual(
            final.usage.input_tokens, 1 + len(self.tokenizer.encode(prompt).ids)
        )

    def test_reports_streamed_token_limits_through_the_official_sdk(self):
        """Returns streamed text and an incomplete response when the token budget is used."""
        sdk, _ = self.sdk()
        with sdk.responses.create(
            model="test-model",
            input="日本語 English",
            max_output_tokens=4,
            temperature=0,
            stream=True,
        ) as stream:
            events = list(stream)
        text = "".join(
            event.delta
            for event in events
            if event.type == "response.output_text.delta"
        )
        self.assertEqual(text, "aaaa")
        self.assertEqual(events[-1].type, "response.incomplete")
        self.assertEqual(events[-1].response.output_text, text)
        self.assertEqual(events[-1].response.usage.output_tokens, 4)

    def test_releases_generation_capacity_when_a_streaming_client_disconnects(self):
        """Accepts the next generation after a streaming client disconnects."""
        sdk, app = self.sdk()
        sdk.responses.create(
            model="test-model", input="hello", max_output_tokens=1, temperature=0
        )
        model = app.state.responses.backend.model
        started, release = threading.Event(), threading.Event()
        original = model.cached_logits

        def delayed(*arguments):
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Generation was not released")
            return original(*arguments)

        with patch.object(model, "cached_logits", side_effect=delayed):
            stream = sdk.responses.create(
                model="test-model",
                input="hello",
                max_output_tokens=4,
                temperature=0,
                stream=True,
            )
            self.assertEqual(next(stream).type, "response.created")
            self.assertTrue(started.wait(timeout=5))
            stream.close()
            release.set()
            deadline = time.monotonic() + 5
            while True:
                try:
                    response = sdk.responses.create(
                        model="test-model",
                        input="hello",
                        max_output_tokens=1,
                        temperature=0,
                    )
                    break
                except openai.RateLimitError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.02)
        self.assertEqual(response.output_text, "a")

    def test_samples_with_the_requested_nucleus_probability(self):
        """Samples from the top nucleus at the requested temperature and probability."""
        response = self.post(top_p=0.001, temperature=1)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["output"][0]["content"][0]["text"], "aaaa")
        self.assertEqual(response.json()["top_p"], 0.001)

    def test_reports_generation_failures_and_accepts_a_subsequent_request(self):
        """Returns JSON or SSE errors on generation failure and accepts the next request."""
        self.post()
        model = self.app.state.responses.backend.model
        with patch.object(
            model, "cached_logits", side_effect=RuntimeError("Private backend details")
        ):
            response = self.post()
            self.assertEqual(response.status_code, 500)
            self.assertEqual(
                response.json()["error"]["message"], "Text generation failed."
            )
            streamed = self.post(stream=True)
            events = [
                json.loads(line[6:])
                for line in streamed.text.splitlines()
                if line.startswith("data: ")
            ]
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertEqual(
                events[-1]["response"]["error"]["message"], "Text generation failed."
            )
        self.assertEqual(self.post().status_code, 200)

    def test_publishes_the_response_contract_and_streaming_content_type(self):
        """Publishes the supported inputs, response schema and streaming content type."""
        specification = self.client.get("/openapi.json").json()
        operation = specification["paths"]["/v1/responses"]["post"]
        schema = specification["components"]["schemas"]["ResponseRequest"]
        self.assertEqual(schema["required"], ["model", "input"])
        self.assertIn("stream", schema["properties"])
        self.assertIn("text/event-stream", operation["responses"]["200"]["content"])
        self.assertIn(
            "model_revision",
            specification["components"]["schemas"]["Response"]["properties"],
        )


class GPUTimeTests(unittest.TestCase):
    def setUp(self):
        self.now = NOON
        self.store = {}
        self.gpu_time = self.limited(300)

    def limited(self, limit):
        return GPUTime(self.store, limit, 60, lambda: self.now)

    def reply(self, seconds):
        started = self.gpu_time.begin()
        self.now += seconds
        self.gpu_time.end(started)

    def spent(self):
        return self.gpu_time.counted(self.now)[0]

    def test_counts_a_reply_until_the_gpu_shuts_down_after_it(self):
        """Counts from the request until the GPU shuts down after the reply."""
        self.reply(24)
        self.assertEqual(self.spent(), 84)

    def test_counts_time_that_replies_share_once(self):
        """Counts the time that replies share once."""
        self.reply(10)
        self.now += 30
        self.reply(10)
        self.assertEqual(self.spent(), 110)

    def test_does_not_count_the_time_the_gpu_is_down(self):
        """Leaves out the time the GPU is down."""
        self.reply(10)
        self.now += 1000
        self.reply(10)
        self.assertEqual(self.spent(), 140)

    def test_counts_a_reply_from_its_start_before_it_ends(self):
        """Counts the time until the GPU would shut down as soon as a reply is asked for."""
        self.gpu_time.begin()
        self.assertEqual(self.spent(), 60)

    def test_refuses_replies_once_the_time_has_run_out_and_lets_the_last_one_finish(self):
        """Counts the reply in progress to its end, then refuses the next, giving the seconds until the next day."""
        self.reply(280)
        self.assertEqual(self.spent(), 340)
        counted = dict(self.store)
        with self.assertRaises(OutOfGPUTime) as refused:
            self.gpu_time.begin()
        self.assertEqual(refused.exception.retry_after, 43200 - 280)
        self.assertEqual(self.store, counted)

    def test_sees_the_time_counted_from_another_container(self):
        """Carries the counted time over to the next container."""
        self.reply(240)
        with self.assertRaises(OutOfGPUTime):
            self.limited(300).begin()

    def test_counts_from_zero_on_the_next_utc_day(self):
        """Accepts replies again when the UTC date changes."""
        self.reply(280)
        self.now = NOON + 43200
        self.gpu_time.begin()
        self.assertEqual(self.spent(), 60)


if __name__ == "__main__":
    unittest.main()
