"""Serve a model in Hugging Face's format on vLLM's server, as pretrain.serving serves our checkpoints.

vLLM writes the prompt with the tokenizer's chat template and tells a reply's
reasoning, text and tool calls apart with the parsers it is started with, so
nothing here depends on which model it serves. Its compiled kernels are cached
under its cache directory, which makes every start after the first shorter.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

PORT = 8000
START_SECONDS = 900
SERVED_NAME = "model"
# vLLM's default sampler needs kernels built with a CUDA toolkit, which the image does not have.
ENVIRONMENT = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def server_command(directory, context, parser):
    return [
        shutil.which("vllm") or "vllm", "serve", str(directory), "--served-model-name", SERVED_NAME,
        "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", str(context), "--gpu-memory-utilization", "0.92",
        "--language-model-only", "--limit-mm-per-prompt", json.dumps({"image": 0, "audio": 0, "video": 0}),
        "--reasoning-parser", parser, "--tool-call-parser", parser, "--enable-auto-tool-choice",
    ]


def chat_form(messages):
    """The conversation and its tools, as a chat API reads them, each tool output answering a call by its id."""
    conversation, tools, unanswered, count = [], [], [], 0
    for message in messages:
        message = dict(message)
        tools += [{"type": "function", "function": tool} for tool in message.pop("tools", ())]
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                identifier = f"call_{count}"
                count += 1
                unanswered.append((identifier, call["name"]))
                calls.append({
                    "id": identifier, "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
                })
            message["tool_calls"] = calls
        if message["role"] == "tool" and unanswered:
            answered = next((entry for entry in unanswered if entry[1] == message.get("name")), unanswered[0])
            unanswered.remove(answered)
            message["tool_call_id"] = answered[0]
        conversation.append(message)
    return conversation, tools


class VLLMModel:
    """A model behind a vLLM server at `url`, which `serve` starts on this machine."""

    def __init__(self, url, context, process=None):
        self.url, self.context, self.process = url, context, process

    @classmethod
    def serve(cls, directory, context, parser="gemma4"):
        process = subprocess.Popen(server_command(directory, context, parser), env={**os.environ, **ENVIRONMENT})
        url = f"http://127.0.0.1:{PORT}"
        deadline = time.monotonic() + START_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM stopped while starting, with exit code {process.returncode}")
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2):
                    return cls(url, context, process)
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        process.kill()
        raise RuntimeError(f"vLLM did not start within {START_SECONDS} seconds")

    def start(self, messages, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
        return VLLMGeneration(self, messages, max_output_tokens, temperature, top_p, reasoning, tools, single_call)

    def post(self, path, body, timeout=600):
        request = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=timeout)

    def close(self):
        """Stop the server, if this object started it, so that what it cached is written out."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()


class VLLMGeneration:
    """A reply's reasoning, text and tool calls as vLLM streams them.

    Iterating yields ("reasoning", text) deltas when the prompt asked for
    reasoning, ("text", text) deltas and a ("call", call) for each call of one
    of `tools`, until the reply ends or reaches its limit. A call of anything
    else, or one that does not parse, stays in the reply's text. With
    `single_call` the reply ends at its first call.
    """

    def __init__(self, backend, messages, max_output_tokens, temperature, top_p, reasoning=False, tools=(), single_call=False):
        self.reasoning, self.tools, self.single_call = reasoning, set(tools), single_call
        conversation, offered = chat_form(messages)
        request = {"model": SERVED_NAME, "messages": conversation, "chat_template_kwargs": {"enable_thinking": bool(reasoning)}}
        if offered:
            request["tools"] = offered
        with backend.post("/tokenize", {**request, "add_generation_prompt": True}, timeout=60) as reply:
            self.input_tokens = json.load(reply)["count"]
        if self.input_tokens + max_output_tokens > backend.context:
            raise ValueError(
                "Input and max_output_tokens exceed the model's context length "
                + str(backend.context)
                + ". Shorten the input or reduce max_output_tokens."
            )
        self.output_tokens = self.reasoning_tokens = self.deltas_read = 0
        self.finished = False
        self.parts = {"reasoning": "", "text": ""}
        if offered:
            request["tool_choice"] = "auto" if self.tools else "none"
        self.reply = backend.post(
            "/v1/chat/completions",
            {**request, "max_tokens": max_output_tokens, "temperature": temperature, "top_p": top_p,
             "stream": True, "stream_options": {"include_usage": True}},
        )
        self.deltas = self._read()

    @property
    def text(self):
        return self.parts["text"]

    @property
    def reasoning_text(self):
        return self.parts["reasoning"]

    def _read(self):
        calls, reason, usage = {}, None, None
        for line in self.reply:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or ():
                reason = choice.get("finish_reason") or reason
                delta = choice.get("delta") or {}
                thought = delta.get("reasoning") or delta.get("reasoning_content")
                if thought:
                    self.reasoning_tokens += 1
                    if self.reasoning:
                        yield from self._write("reasoning", thought)
                if delta.get("content"):
                    self.deltas_read += 1
                    yield from self._write("text", delta["content"])
                for piece in delta.get("tool_calls") or ():
                    index = piece.get("index", 0)
                    # A later call means the earlier ones are whole.
                    for earlier in sorted(position for position in calls if position < index):
                        yield from self._finish_call(calls.pop(earlier))
                        if self.finished:
                            return
                    entry = calls.setdefault(index, {"name": "", "arguments": ""})
                    function = piece.get("function") or {}
                    entry["name"] += function.get("name") or ""
                    entry["arguments"] += function.get("arguments") or ""
        for position in sorted(calls):
            yield from self._finish_call(calls[position])
            if self.finished:
                return
        self.finished = reason in ("stop", "tool_calls")
        if usage:
            self.output_tokens = usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens = details.get("reasoning_tokens") or self.reasoning_tokens
        else:
            self.output_tokens = self.reasoning_tokens + self.deltas_read

    def _finish_call(self, entry):
        try:
            arguments = json.loads(entry["arguments"] or "{}")
        except ValueError:
            arguments = None
        if entry["name"] in self.tools and isinstance(arguments, dict):
            yield "call", {"name": entry["name"], "arguments": arguments}
            if self.single_call:
                self.finished = True
        else:
            yield from self._write("text", json.dumps({"name": entry["name"], "arguments": arguments or entry["arguments"]}, ensure_ascii=False))

    def _write(self, part, text):
        self.parts[part] += text
        yield part, text

    def __iter__(self):
        return self.deltas

    def close(self):
        """Stop reading, which makes the server stop writing."""
        self.deltas.close()
        self.reply.close()
