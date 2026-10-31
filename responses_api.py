import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.background import BackgroundTask

from pretrain.remote import OutOfGPUTime


class APIError(Exception):
    def __init__(
        self,
        message,
        param=None,
        code="invalid_parameter",
        status=400,
        kind="invalid_request_error",
        retry_after=1,
    ):
        self.message, self.param, self.code = message, param, code
        self.status, self.kind, self.retry_after = status, kind, retry_after

    def response(self):
        headers = {"Retry-After": str(self.retry_after)} if self.status == 429 else None
        return JSONResponse(
            status_code=self.status,
            content={
                "error": {
                    "message": self.message,
                    "type": self.kind,
                    "param": self.param,
                    "code": self.code,
                }
            },
            headers=headers,
        )


def validation_error(error):
    first = error.errors()[0]
    location = [str(part) for part in first["loc"] if part != "body"]
    code = {
        "extra_forbidden": "unknown_parameter",
        "missing": "missing_required_parameter",
        "json_invalid": "invalid_json",
    }.get(first["type"], "invalid_parameter")
    return APIError(first["msg"], ".".join(location) or None, code).response()


class InputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TextPart(InputSchema):
    type: Literal["input_text", "output_text"]
    text: str
    annotations: list = Field(default_factory=list, max_length=0)
    logprobs: list = Field(default_factory=list, max_length=0)
    # The SDK sends back the parsed value of a streamed response's text, null without a structured format.
    parsed: None = None


class InputMessage(InputSchema):
    role: Literal["user", "assistant", "system", "developer"]
    content: str | list[TextPart]
    type: Literal["message"] = "message"
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class InputReasoning(InputSchema):
    """A reasoning item from an earlier response; earlier reasoning is left out of the prompt."""

    type: Literal["reasoning"]
    id: str | None = None
    summary: list = Field(default_factory=list)
    content: list | None = None
    encrypted_content: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class InputFunctionCall(InputSchema):
    """A tool call from an earlier response, which joins the assistant turn it follows."""

    type: Literal["function_call"]
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class InputFunctionCallOutput(InputSchema):
    """What an earlier tool call returned."""

    type: Literal["function_call_output"]
    call_id: str = Field(min_length=1)
    output: str | list[TextPart]
    id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None


class FunctionTool(InputSchema):
    type: Literal["function"]
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str | None = None
    parameters: dict | None = None
    strict: bool | None = None

    def definition(self):
        """The tool as the system turn lists it: its name, description and arguments' schema."""
        fields = (("name", self.name), ("description", self.description), ("parameters", self.parameters))
        return {key: value for key, value in fields if value is not None}


class ReasoningOptions(InputSchema):
    effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class TextFormat(InputSchema):
    type: Literal["text"] = "text"


class TextOptions(InputSchema):
    format: TextFormat = Field(default_factory=TextFormat)


class ResponseRequest(InputSchema):
    model: str = Field(min_length=1, max_length=128)
    input: str | list[InputMessage | InputReasoning | InputFunctionCall | InputFunctionCallOutput]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    stream: bool | None = False
    store: bool | None = Field(
        default=False,
        description="Only false is supported. Responses are not stored on the server.",
    )
    metadata: (
        dict[
            Annotated[str, StringConstraints(max_length=64)],
            Annotated[str, StringConstraints(max_length=512)],
        ]
        | None
    ) = Field(default=None, max_length=16)
    text: TextOptions | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["auto", "none"] | None = "auto"
    parallel_tool_calls: bool | None = True
    previous_response_id: str | None = None
    background: bool | None = False
    truncation: Literal["disabled"] | None = "disabled"
    include: list[str] | None = None
    reasoning: ReasoningOptions | None = None

    def reasons(self):
        """Whether the reply reasons before it answers, as low or higher effort asks."""
        return self.reasoning is not None and self.reasoning.effort in ("low", "medium", "high")

    def callable_tools(self):
        """The names of the tools the reply may call, none when tool_choice is none."""
        return [tool.name for tool in self.tools or ()] if self.tool_choice != "none" else []

    def messages(self):
        """The conversation's turns, from which the model writes its own prompt."""
        for name in ("store", "previous_response_id", "background", "include"):
            if getattr(self, name):
                raise APIError(
                    "This server supports stateless text generation; "
                    + name
                    + " is not supported.",
                    name,
                    "unsupported_parameter",
                )
        tools = self.tools or []
        if len({tool.name for tool in tools}) < len(tools):
            raise APIError("Give each tool a different name.", "tools")
        messages, calls = [], {}
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        inputs = (
            [InputMessage(role="user", content=self.input)]
            if isinstance(self.input, str)
            else self.input
        )
        for index, message in enumerate(inputs):
            if isinstance(message, InputReasoning):
                continue
            if isinstance(message, InputFunctionCall):
                try:
                    arguments = json.loads(message.arguments)
                except ValueError:
                    arguments = None
                if not isinstance(arguments, dict):
                    raise APIError(
                        "Give a function call's arguments as a JSON object.",
                        "input." + str(index) + ".arguments",
                    )
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": ""})
                messages[-1].setdefault("tool_calls", []).append({"name": message.name, "arguments": arguments})
                calls[message.call_id] = message.name
                continue
            if isinstance(message, InputFunctionCallOutput):
                if message.call_id not in calls:
                    raise APIError(
                        "No function call before this output has its call_id.",
                        "input." + str(index) + ".call_id",
                    )
                if isinstance(message.output, str):
                    output = message.output
                elif any(part.type != "input_text" for part in message.output):
                    raise APIError("Use input_text for function call outputs.", "input." + str(index) + ".output")
                else:
                    output = "".join(part.text for part in message.output)
                messages.append({"role": "tool", "content": output, "name": calls[message.call_id]})
                continue
            if isinstance(message.content, str):
                content = message.content
            else:
                for part in message.content:
                    if part.type == "output_text" and message.role != "assistant":
                        raise APIError(
                            "Use input_text for user, system and developer messages.",
                            "input." + str(index) + ".content",
                        )
                content = "".join(part.text for part in message.content)
            messages.append(
                {
                    "role": "system" if message.role == "developer" else message.role,
                    "content": content,
                }
            )
        if tools:
            # The system turn that opens the conversation lists the tools, as in training.
            if not messages or messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
            messages[0]["tools"] = [tool.definition() for tool in tools]
        return messages


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: list = Field(default_factory=list)
    logprobs: list = Field(default_factory=list)


class OutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"]
    content: list[OutputText] = Field(default_factory=list)


class ReasoningText(BaseModel):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str = ""


class OutputReasoning(BaseModel):
    id: str
    type: Literal["reasoning"] = "reasoning"
    status: Literal["in_progress", "completed", "incomplete"]
    summary: list = Field(default_factory=list)
    content: list[ReasoningText] = Field(default_factory=list)


class OutputFunctionCall(BaseModel):
    id: str
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str = ""
    status: Literal["in_progress", "completed", "incomplete"]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"cached_tokens": 0, "cache_write_tokens": 0}
    )
    output_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"reasoning_tokens": 0}
    )


class Response(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int | None = None
    status: Literal["in_progress", "completed", "incomplete", "failed"] = "in_progress"
    error: dict | None = None
    incomplete_details: dict | None = None
    model: str
    model_revision: str
    instructions: str | None = None
    max_output_tokens: int
    output: list[OutputReasoning | OutputMessage | OutputFunctionCall] = Field(default_factory=list)
    usage: Usage | None = None
    store: Literal[False] = False
    background: Literal[False] = False
    previous_response_id: None = None
    temperature: float
    top_p: float
    text: TextOptions = Field(default_factory=TextOptions)
    metadata: dict[str, str] = Field(default_factory=dict)
    tools: list = Field(default_factory=list)
    tool_choice: Literal["auto", "none"] = "auto"
    parallel_tool_calls: bool = True
    truncation: Literal["disabled"] = "disabled"
    reasoning: dict = Field(default_factory=lambda: {"effort": None, "summary": None})
    service_tier: Literal["default"] = "default"


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "bilingual-v2-1b"
    checkpoint: Path = Path(__file__).parent / "runs/bilingual-v2-1b/model.pt"
    hub_repo: str | None = None
    hub_run_id: str = "bilingual-v2-1b"
    hub_revision: str | None = None
    cache_dir: Path = Path(__file__).parent / "runs/hub"
    device: str = "cpu"
    threads: int = 1
    max_output_tokens: int = 512
    top_p: float = 1.0
    api_key: str | None = None

    def __post_init__(self):
        if not self.model_id or self.threads <= 0 or self.max_output_tokens <= 0:
            raise ValueError("Provide a model ID and positive thread and output limits")
        if self.hub_repo and not re.fullmatch(r"[a-f0-9]{40}", self.hub_revision or ""):
            raise ValueError("Pin HF_MODEL_REVISION to the model's full commit SHA")

    @classmethod
    def from_env(cls):
        return cls(
            model_id=os.environ.get("MODEL_ID", cls.model_id),
            checkpoint=Path(os.environ.get("MODEL_CHECKPOINT", cls.checkpoint)),
            hub_repo=os.environ.get("HF_MODEL_REPO"),
            hub_run_id=os.environ.get("HF_MODEL_RUN_ID", cls.hub_run_id),
            hub_revision=os.environ.get("HF_MODEL_REVISION"),
            cache_dir=Path(os.environ.get("MODEL_CACHE_DIR", cls.cache_dir)),
            device=os.environ.get("MODEL_DEVICE", "cpu"),
            threads=int(os.environ.get("MODEL_THREADS", "1")),
            max_output_tokens=int(os.environ.get("MODEL_MAX_OUTPUT_TOKENS", "512")),
            api_key=os.environ.get("LANGUAGE_MODEL_API_KEY"),
        )


class ResponseService:
    def __init__(self, settings, backend=None):
        self.settings = settings
        self.capacity = threading.BoundedSemaphore(1)
        # A model given here, such as one sampling in another container, names its own revision.
        self.backend = backend
        self.revision = None if backend is None else backend.revision

    def authorize(self, request):
        if self.settings.api_key:
            expected = ("Bearer " + self.settings.api_key).encode()
            actual = request.headers.get("Authorization", "").encode()
            if not secrets.compare_digest(actual, expected):
                raise APIError(
                    "Provide a valid API key using Bearer authentication.",
                    code="invalid_api_key",
                    status=401,
                    kind="authentication_error",
                )

    def load(self):
        if self.backend is not None:
            return
        from pretrain.serving import TextModel

        checkpoint = self.settings.checkpoint
        if self.settings.hub_repo:
            from pretrain.hub import download_checkpoint

            restored = download_checkpoint(
                self.settings.hub_repo,
                self.settings.hub_run_id,
                self.settings.cache_dir,
                revision=self.settings.hub_revision,
            )
            checkpoint = Path(restored["checkpoint"])
            revision = restored["revision"]
        else:
            with checkpoint.open("rb") as source:
                revision = hashlib.file_digest(source, "sha256").hexdigest()
        backend = TextModel(checkpoint, self.settings.device, self.settings.threads)
        self.backend, self.revision = backend, revision

    def start(self, payload):
        messages = payload.messages()
        if payload.model != self.settings.model_id:
            raise APIError(
                "The requested model is not available. Check GET /v1/models.",
                "model",
                "model_not_found",
                404,
            )
        limit = payload.max_output_tokens or min(128, self.settings.max_output_tokens)
        if limit > self.settings.max_output_tokens:
            raise APIError(
                "max_output_tokens must be at most "
                + str(self.settings.max_output_tokens),
                "max_output_tokens",
            )
        if not self.capacity.acquire(blocking=False):
            raise APIError(
                "Generation is busy. Try again shortly.",
                code="rate_limit_exceeded",
                status=429,
                kind="rate_limit_error",
            )
        try:
            try:
                self.load()
            except Exception as error:  # noqa: BLE001 - Do not expose private source URLs.
                logging.getLogger(__name__).error(
                    "Text model loading failed (%s)", type(error).__name__
                )
                raise APIError(
                    "The text model is unavailable. Try again later.",
                    code="model_unavailable",
                    status=503,
                    kind="server_error",
                ) from None
            temperature = (
                payload.temperature if payload.temperature is not None else 1.0
            )
            top_p = payload.top_p if payload.top_p is not None else self.settings.top_p
            try:
                generation = self.backend.start(
                    messages,
                    limit,
                    temperature,
                    top_p,
                    payload.reasons(),
                    payload.callable_tools(),
                    payload.parallel_tool_calls is False,
                )
            except ValueError as error:
                raise APIError(str(error), "input", "context_length_exceeded") from None
            except OutOfGPUTime as error:
                raise APIError(
                    "The model has used up its GPU time for today. Try again after 00:00 UTC.",
                    code="insufficient_quota",
                    status=429,
                    kind="insufficient_quota",
                    retry_after=error.retry_after,
                ) from None
            response = Response(
                id="resp_" + secrets.token_hex(16),
                created_at=int(time.time()),
                model=payload.model,
                model_revision=self.revision,
                instructions=payload.instructions,
                max_output_tokens=limit,
                temperature=temperature,
                top_p=top_p,
                metadata=payload.metadata or {},
                tools=[tool.model_dump() for tool in payload.tools or ()],
                tool_choice=payload.tool_choice or "auto",
                parallel_tool_calls=payload.parallel_tool_calls is not False,
                reasoning=(payload.reasoning or ReasoningOptions()).model_dump(),
            )
            return ResponseSession(generation, response, self.capacity)
        except BaseException:
            self.capacity.release()
            raise


class ResponseSession:
    def __init__(self, generation, response, capacity):
        self.generation, self.response, self.capacity = generation, response, capacity
        # A reply that reasons first returns its reasoning as an item before the others.
        self.thought = (
            OutputReasoning(id="rs_" + secrets.token_hex(16), status="in_progress")
            if generation.reasoning
            else None
        )
        self.items = [] if self.thought is None else [self.thought]
        self.current = self.thought
        self.closed = False

    def wait(self):
        """Return once the model has started the reply, reporting an input it refuses."""
        try:
            self.generation.wait()
        except ValueError as error:
            raise APIError(str(error), "input", "context_length_exceeded") from None

    def updates(self):
        """Fill the output items as the reply is generated, yielding the stream's events for them.

        The reply's text goes to a message opened when the text starts. Each
        tool call is an item of its own, and text after a call opens another
        message. A reply with neither still returns an empty message.
        """
        if self.thought is not None:
            yield "response.output_item.added", {"output_index": 0, "item": self.thought.model_dump()}
        for phase, delta in self.generation:
            if phase == "reasoning":
                yield "response.reasoning_text.delta", {**self.address(self.thought), "delta": delta}
            elif phase == "text":
                if not isinstance(self.current, OutputMessage):
                    yield from self.close_item("completed")
                    yield from self.open_message()
                self.current.content[0].text += delta
                yield "response.output_text.delta", {**self.address(self.current), "delta": delta, "logprobs": []}
            else:
                yield from self.close_item("completed")
                yield from self.add_call(delta)
        status = "completed" if self.generation.finished else "incomplete"
        yield from self.close_item(status)
        if all(item is self.thought for item in self.items):
            yield from self.open_message()
            yield from self.close_item(status)

    def address(self, item):
        return {"item_id": item.id, "output_index": self.items.index(item), "content_index": 0}

    def add(self, item):
        self.items.append(item)
        return {"output_index": len(self.items) - 1, "item": item.model_dump()}

    def open_message(self):
        self.current = OutputMessage(id="msg_" + secrets.token_hex(16), status="in_progress")
        yield "response.output_item.added", self.add(self.current)
        self.current.content = [OutputText()]
        yield "response.content_part.added", {**self.address(self.current), "part": OutputText().model_dump()}

    def close_item(self, status):
        """Finish the item the reply is writing, if any."""
        item, self.current = self.current, None
        if item is None:
            return
        item.status = status
        if item is self.thought:
            item.content = [ReasoningText(text=self.generation.reasoning_text)]
            yield "response.reasoning_text.done", {**self.address(item), "text": self.generation.reasoning_text}
        else:
            yield "response.output_text.done", {**self.address(item), "text": item.content[0].text, "logprobs": []}
            yield "response.content_part.done", {**self.address(item), "part": item.content[0].model_dump()}
        yield "response.output_item.done", {"output_index": self.items.index(item), "item": item.model_dump()}

    def add_call(self, call):
        item = OutputFunctionCall(
            id="fc_" + secrets.token_hex(16),
            call_id="call_" + secrets.token_hex(16),
            name=call["name"],
            status="in_progress",
        )
        added = self.add(item)
        address = {"item_id": item.id, "output_index": added["output_index"]}
        yield "response.output_item.added", added
        arguments = json.dumps(call["arguments"], ensure_ascii=False)
        yield "response.function_call_arguments.delta", {**address, "delta": arguments}
        item.arguments, item.status = arguments, "completed"
        yield "response.function_call_arguments.done", {**address, "arguments": arguments}
        yield "response.output_item.done", {"output_index": address["output_index"], "item": item.model_dump()}

    def record(self):
        """Put the output items in the response and count its tokens."""
        self.response.output = list(self.items)
        self.response.usage = Usage(
            input_tokens=self.generation.input_tokens,
            output_tokens=self.generation.output_tokens,
            total_tokens=self.generation.input_tokens + self.generation.output_tokens,
            output_tokens_details={"reasoning_tokens": self.generation.reasoning_tokens},
        )

    def finish(self):
        status = "completed" if self.generation.finished else "incomplete"
        self.record()
        self.response.status = status
        self.response.completed_at = (
            int(time.time()) if self.generation.finished else None
        )
        if not self.generation.finished:
            self.response.incomplete_details = {"reason": "max_output_tokens"}
        return self.response

    def fail(self, code="server_error", message="Text generation failed."):
        for item in self.items:
            if item.status == "in_progress":
                item.status = "incomplete"
        if self.thought is not None:
            self.thought.content = [ReasoningText(text=self.generation.reasoning_text)]
        self.record()
        self.response.status = "failed"
        self.response.error = {"code": code, "message": message}
        return self.response

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.generation.close()
            finally:
                self.capacity.release()

    async def events(self):
        sequence = 0

        def event(kind, **fields):
            nonlocal sequence
            payload = {"type": kind, "sequence_number": sequence, **fields}
            sequence += 1
            return (
                "event: "
                + kind
                + "\ndata: "
                + json.dumps(payload, ensure_ascii=False)
                + "\n\n"
            )

        try:
            for kind in ("response.created", "response.in_progress"):
                yield event(kind, response=self.response.model_dump())
            # The response has now started, so however long the model takes to
            # start, the request stays within the time a web request may wait.
            await anyio.to_thread.run_sync(self.wait)
            updates = self.updates()
            sentinel = object()
            while True:
                # Shield an active model step so cancellation cannot race with closing it.
                update = await anyio.to_thread.run_sync(next, updates, sentinel)
                if update is sentinel:
                    break
                kind, fields = update
                yield event(kind, **fields)
            response = self.finish()
            yield event("response." + response.status, response=response.model_dump())
        except APIError as error:
            yield event("response.failed", response=self.fail(error.code, error.message).model_dump())
        except Exception:  # noqa: BLE001 - Finish the SSE protocol on a backend failure.
            yield event("response.failed", response=self.fail().model_dump())
        finally:
            # Closing a reply can wait on the network, which would hold up the event loop.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(self.close)


def install_routes(app: FastAPI, settings=None, backend=None):
    service = ResponseService(settings or ModelSettings.from_env(), backend)
    app.state.responses = service
    access_key = HTTPBearer(
        scheme_name="LanguageModelAPIKey",
        description="言語モデルAPIのアクセスキーを入力してください。",
        bearerFormat="API key",
        auto_error=False,
    )
    authentication = [Depends(access_key)] if service.settings.api_key else []

    @app.exception_handler(APIError)
    async def api_error(request, error):
        return error.response()

    @app.get(
        "/v1/models",
        summary="List available text models",
        dependencies=authentication,
    )
    def models(request: Request):
        service.authorize(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": service.settings.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post(
        "/v1/responses",
        response_model=Response,
        dependencies=authentication,
        summary="Create a text response",
        description="Stateless text generation with JSON or Responses SSE events. Supports string inputs, text messages, instructions and function tools. Use store=false. Images, built-in tools and stored conversations are not supported.",
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/json": {
                        "example": {
                            "model": service.settings.model_id,
                            "input": [
                                {"role": "user", "content": "日本の首都はどこですか？"}
                            ],
                            "max_output_tokens": min(
                                64, service.settings.max_output_tokens
                            ),
                            "temperature": 0,
                            "store": False,
                        }
                    }
                }
            }
        },
        responses={
            200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}
        },
    )
    def create_response(payload: ResponseRequest, request: Request):
        service.authorize(request)
        session = service.start(payload)
        if payload.stream:
            return StreamingResponse(
                session.events(),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "x-request-id": session.response.id,
                },
                background=BackgroundTask(session.close),
            )
        try:
            session.wait()
            for _ in session.updates():
                pass
            return JSONResponse(
                session.finish().model_dump(),
                headers={"x-request-id": session.response.id},
            )
        except APIError:
            raise
        except Exception:  # noqa: BLE001 - Keep backend failures in the API error format.
            raise APIError(
                "Text generation failed.",
                code="server_error",
                status=500,
                kind="server_error",
            ) from None
        finally:
            session.close()
