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

from pretrain.conversation import chat_prompt


class APIError(Exception):
    def __init__(
        self,
        message,
        param=None,
        code="invalid_parameter",
        status=400,
        kind="invalid_request_error",
    ):
        self.message, self.param, self.code = message, param, code
        self.status, self.kind = status, kind

    def response(self):
        headers = {"Retry-After": "1"} if self.status == 429 else None
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


class ReasoningOptions(InputSchema):
    effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class TextFormat(InputSchema):
    type: Literal["text"] = "text"


class TextOptions(InputSchema):
    format: TextFormat = Field(default_factory=TextFormat)


class ResponseRequest(InputSchema):
    model: str = Field(min_length=1, max_length=128)
    input: str | list[InputMessage | InputReasoning]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
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
    tools: list[dict] | None = None
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

    def prompt(self):
        for name in ("store", "previous_response_id", "background", "tools", "include"):
            if getattr(self, name):
                raise APIError(
                    "This server supports stateless text generation; "
                    + name
                    + " is not supported.",
                    name,
                    "unsupported_parameter",
                )
        messages = []
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
        return chat_prompt(messages, self.reasons())


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
    output: list[OutputReasoning | OutputMessage] = Field(default_factory=list)
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
    def __init__(self, settings):
        self.settings = settings
        self.capacity = threading.BoundedSemaphore(1)
        self.backend = None
        self.revision = None

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
        prompt = payload.prompt()
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
            top_p = payload.top_p if payload.top_p is not None else 1.0
            try:
                generation = self.backend.start(prompt, limit, temperature, top_p, payload.reasons())
            except ValueError as error:
                raise APIError(str(error), "input", "context_length_exceeded") from None
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
        self.message = OutputMessage(
            id="msg_" + secrets.token_hex(16), status="in_progress"
        )
        # A reply that reasons first returns its reasoning as an item before the message.
        self.thought = (
            OutputReasoning(id="rs_" + secrets.token_hex(16), status="in_progress")
            if generation.reasoning
            else None
        )
        self.closed = False

    def close_thought(self, status):
        self.thought.status = status
        self.thought.content = [ReasoningText(text=self.generation.reasoning_text)]

    def record(self, status):
        """Put the generated parts in the response's output and count its tokens."""
        self.message.status = status
        self.message.content = [OutputText(text=self.generation.text)]
        if self.thought is not None:
            self.close_thought("completed" if self.generation.text or self.generation.finished else status)
        self.response.output = [item for item in (self.thought, self.message) if item is not None]
        self.response.usage = Usage(
            input_tokens=self.generation.input_tokens,
            output_tokens=self.generation.output_tokens,
            total_tokens=self.generation.input_tokens + self.generation.output_tokens,
            output_tokens_details={"reasoning_tokens": self.generation.reasoning_tokens},
        )

    def finish(self):
        status = "completed" if self.generation.finished else "incomplete"
        self.record(status)
        self.response.status = status
        self.response.completed_at = (
            int(time.time()) if self.generation.finished else None
        )
        if not self.generation.finished:
            self.response.incomplete_details = {"reason": "max_output_tokens"}
        return self.response

    def fail(self):
        self.record("incomplete")
        self.response.status = "failed"
        self.response.error = {
            "code": "server_error",
            "message": "Text generation failed.",
        }
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

        index = 0 if self.thought is None else 1
        address = {"item_id": self.message.id, "output_index": index, "content_index": 0}

        def open_message():
            yield event(
                "response.output_item.added",
                output_index=index,
                item=self.message.model_dump(),
            )
            yield event(
                "response.content_part.added", **address, part=OutputText().model_dump()
            )

        def thought_done():
            thinking = {"item_id": self.thought.id, "output_index": 0, "content_index": 0}
            yield event(
                "response.reasoning_text.done", **thinking, text=self.generation.reasoning_text
            )
            yield event(
                "response.output_item.done", output_index=0, item=self.thought.model_dump()
            )

        try:
            for kind in ("response.created", "response.in_progress"):
                yield event(kind, response=self.response.model_dump())
            if self.thought is None:
                for line in open_message():
                    yield line
            else:
                yield event(
                    "response.output_item.added",
                    output_index=0,
                    item=self.thought.model_dump(),
                )
            answering = self.thought is None
            iterator = iter(self.generation)
            sentinel = object()
            while True:
                # Shield an active model step so cancellation cannot race with closing it.
                item = await anyio.to_thread.run_sync(next, iterator, sentinel)
                if item is sentinel:
                    break
                phase, delta = item
                if phase == "reasoning":
                    yield event(
                        "response.reasoning_text.delta",
                        item_id=self.thought.id,
                        output_index=0,
                        content_index=0,
                        delta=delta,
                    )
                    continue
                if not answering:
                    self.close_thought("completed")
                    for line in (*thought_done(), *open_message()):
                        yield line
                    answering = True
                yield event(
                    "response.output_text.delta", **address, delta=delta, logprobs=[]
                )
            response = self.finish()
            if not answering:
                for line in (*thought_done(), *open_message()):
                    yield line
            yield event(
                "response.output_text.done",
                **address,
                text=self.generation.text,
                logprobs=[],
            )
            yield event(
                "response.content_part.done",
                **address,
                part=self.message.content[0].model_dump(),
            )
            yield event(
                "response.output_item.done",
                output_index=index,
                item=self.message.model_dump(),
            )
            yield event("response." + response.status, response=response.model_dump())
        except Exception:  # noqa: BLE001 - Finish the SSE protocol on a backend failure.
            yield event("response.failed", response=self.fail().model_dump())
        finally:
            self.close()


def install_routes(app: FastAPI, settings=None):
    service = ResponseService(settings or ModelSettings.from_env())
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
        description="Stateless text generation with JSON or Responses SSE events. Supports string inputs, text messages and instructions. Use store=false. Images, tools and stored conversations are not supported.",
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
            for _ in session.generation:
                pass
            return JSONResponse(
                session.finish().model_dump(),
                headers={"x-request-id": session.response.id},
            )
        except Exception:  # noqa: BLE001 - Keep backend failures in the API error format.
            raise APIError(
                "Text generation failed.",
                code="server_error",
                status=500,
                kind="server_error",
            ) from None
        finally:
            session.close()
