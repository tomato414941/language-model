import json
import pickle
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

from pretrain.remote import GPUTime, RemoteTextModel, relay
from pretrain.vllm_serving import VLLMModel, chat_form
from responses_api import ModelSettings
from server import create_app

USER = [{"role": "user", "content": "天気は？"}]
WEATHER = {"name": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
OFFERING = [{"role": "system", "content": "", "tools": [WEATHER]}, *USER]


def delta(finish_reason=None, **fields):
    return {"choices": [{"delta": fields, "finish_reason": finish_reason}]}


def tool_call(index, name=None, arguments=""):
    return {"index": index, "function": {"name": name, "arguments": arguments}}


class StandIn(BaseHTTPRequestHandler):
    """A stand-in for vLLM's server, which counts words and streams the chunks it is given."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body))
        if self.path == "/tokenize":
            count = sum(len(message.get("content", "")) for message in body["messages"])
            self.reply(json.dumps({"count": count}).encode(), "application/json")
            return
        lines = [f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in self.server.chunks]
        self.reply(("".join(lines) + "data: [DONE]\n\n").encode(), "text/event-stream")

    def reply(self, data, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *arguments):
        pass


class VLLMModelTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), StandIn)
        self.server.requests, self.server.chunks = [], []
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.model = VLLMModel(f"http://127.0.0.1:{self.server.server_port}", context=256)

    def reply(self, messages=USER, limit=32, **options):
        generation = self.model.start(messages, limit, 1.0, 0.95, **options)
        deltas = list(generation)
        generation.close()
        return generation, deltas

    def test_returns_the_text_and_counts_the_tokens_of_a_finished_reply(self):
        """Returns a reply's text, with its token counts from the server."""
        self.server.chunks = [
            delta(content="雨"), delta(content="です。"), delta("stop"),
            {"choices": [], "usage": {"completion_tokens": 5}},
        ]
        generation, deltas = self.reply()
        self.assertEqual(deltas, [("text", "雨"), ("text", "です。")])
        self.assertEqual(generation.text, "雨です。")
        self.assertEqual(generation.input_tokens, 4)
        self.assertEqual(generation.output_tokens, 5)
        self.assertTrue(generation.finished)

    def test_leaves_a_reply_unfinished_at_its_limit(self):
        """Leaves a reply that reaches its limit unfinished."""
        self.server.chunks = [delta(content="雨"), delta("length")]
        generation, _ = self.reply()
        self.assertFalse(generation.finished)

    def test_asks_for_the_settings_of_the_reply(self):
        """Asks the server for the reply's length, sampling and reasoning, as a stream with its usage."""
        self.server.chunks = [delta("stop")]
        self.reply(limit=16, reasoning=True)
        path, body = self.server.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual((body["max_tokens"], body["temperature"], body["top_p"]), (16, 1.0, 0.95))
        self.assertTrue(body["stream"])
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})

    def test_separates_the_reasoning_from_the_text(self):
        """Returns reasoning and text apart, counting the tokens of the reasoning."""
        self.server.chunks = [delta(reasoning="考える。"), delta(content="雨です。"), delta("stop")]
        generation, deltas = self.reply(reasoning=True)
        self.assertEqual(deltas, [("reasoning", "考える。"), ("text", "雨です。")])
        self.assertEqual(generation.reasoning_text, "考える。")
        self.assertEqual(generation.reasoning_tokens, 1)

    def test_leaves_out_reasoning_the_request_did_not_ask_for(self):
        """Leaves out the reasoning of a reply when the request did not ask for it."""
        self.server.chunks = [delta(reasoning="考える。"), delta(content="雨です。"), delta("stop")]
        generation, deltas = self.reply()
        self.assertEqual(deltas, [("text", "雨です。")])
        self.assertEqual(generation.reasoning_text, "")

    def test_refuses_an_input_beyond_the_context_without_asking_for_a_reply(self):
        """Refuses an input that leaves no room for the reply, before asking for one."""
        with self.assertRaisesRegex(ValueError, "context length 256"):
            self.model.start(USER, 256, 1.0, 1.0)
        self.assertEqual([path for path, _ in self.server.requests], ["/tokenize"])

    def test_returns_calls_of_the_offered_tools_when_they_are_whole(self):
        """Returns each call of an offered tool once its arguments are whole, between the reply's text."""
        self.server.chunks = [
            delta(content="見ます。"),
            delta(tool_calls=[tool_call(0, "weather", '{"city": ')]),
            delta(tool_calls=[tool_call(0, None, '"東京"}')]),
            delta(tool_calls=[tool_call(1, "weather", '{"city": "大阪"}')]),
            delta("tool_calls"),
        ]
        generation, deltas = self.reply(OFFERING, tools=["weather"])
        self.assertEqual(
            deltas,
            [("text", "見ます。"), ("call", {"name": "weather", "arguments": {"city": "東京"}}), ("call", {"name": "weather", "arguments": {"city": "大阪"}})],
        )
        self.assertTrue(generation.finished)
        self.assertEqual(self.server.requests[-1][1]["tool_choice"], "auto")

    def test_keeps_calls_of_anything_else_or_that_do_not_parse_in_the_text(self):
        """Keeps a call of a tool not offered, or whose arguments do not parse, in the reply's text."""
        for name, arguments in (("clock", '{"zone": "JST"}'), ("weather", '{"city": ')):
            with self.subTest(name=name):
                self.server.chunks = [delta(tool_calls=[tool_call(0, name, arguments)]), delta("tool_calls")]
                generation, deltas = self.reply(OFFERING, tools=["weather"])
                self.assertEqual({phase for phase, _ in deltas}, {"text"})
                self.assertIn(name, generation.text)

    def test_ends_the_reply_at_its_first_call_when_asked(self):
        """Ends the reply at its first call when the request allows only one, and calls it finished."""
        self.server.chunks = [
            delta(tool_calls=[tool_call(0, "weather", '{"city": "東京"}')]),
            delta(tool_calls=[tool_call(1, "weather", '{"city": "大阪"}')]),
            delta("tool_calls"),
        ]
        generation, deltas = self.reply(OFFERING, tools=["weather"], single_call=True)
        self.assertEqual(deltas, [("call", {"name": "weather", "arguments": {"city": "東京"}})])
        self.assertTrue(generation.finished)

    def test_offers_tools_without_letting_the_reply_call_them_when_none_may_be_called(self):
        """Lists the tools but stops the reply calling them when tool_choice leaves none callable."""
        self.server.chunks = [delta("stop")]
        self.reply(OFFERING)
        body = self.server.requests[-1][1]
        self.assertEqual(body["tools"], [{"type": "function", "function": WEATHER}])
        self.assertEqual(body["tool_choice"], "none")

    def test_gives_each_tool_output_the_id_of_the_call_it_answers(self):
        """Gives earlier calls their ids and JSON arguments, and each tool output the id of its call."""
        conversation, tools = chat_form([
            *OFFERING,
            {"role": "assistant", "content": "", "tool_calls": [{"name": "weather", "arguments": {"city": "東京"}}, {"name": "now", "arguments": {}}]},
            {"role": "tool", "content": "12時", "name": "now"},
            {"role": "tool", "content": "雨", "name": "weather"},
        ])
        self.assertEqual(tools, [{"type": "function", "function": WEATHER}])
        calls = conversation[2]["tool_calls"]
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "東京"})
        self.assertEqual(conversation[3]["tool_call_id"], calls[1]["id"])
        self.assertEqual(conversation[4]["tool_call_id"], calls[0]["id"])

    def test_serves_the_reply_and_its_reasoning_through_the_api(self):
        """Serves a reply and its reasoning through the API, as JSON and as a stream."""

        def sample(*arguments):
            for record in relay(self.model, *arguments):
                yield pickle.loads(pickle.dumps(record))

        backend = RemoteTextModel(sample, "f" * 40, GPUTime({}, 3600, 60))
        settings = ModelSettings(model_id="test-model", max_output_tokens=32, top_p=0.95)
        self.server.chunks = [
            delta(reasoning="考える。"), delta(content="雨です。"), delta("stop"),
            {"choices": [], "usage": {"completion_tokens": 7, "completion_tokens_details": {"reasoning_tokens": 3}}},
        ]
        with TestClient(create_app(response_settings=settings, backend=backend)) as client:
            request = {"model": "test-model", "input": "天気は？", "max_output_tokens": 32, "reasoning": {"effort": "low"}}
            response = client.post("/v1/responses", json=request).json()
            self.assertEqual(response["status"], "completed")
            self.assertEqual([item["type"] for item in response["output"]], ["reasoning", "message"])
            self.assertEqual(response["output"][1]["content"][0]["text"], "雨です。")
            self.assertEqual(response["usage"]["output_tokens"], 7)
            self.assertEqual(response["usage"]["output_tokens_details"]["reasoning_tokens"], 3)
            with client.stream("POST", "/v1/responses", json={**request, "stream": True}) as stream:
                events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
            self.assertEqual(events[-1]["type"], "response.completed")
            self.assertEqual(
                "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta"), "雨です。"
            )


if __name__ == "__main__":
    unittest.main()
