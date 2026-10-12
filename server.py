"""Serve locally trained models with FastAPI and the Responses API."""

import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from model import load_model
from responses_api import (
    APIError,
    install_routes,
)
from responses_api import (
    validation_error as response_validation_error,
)

CHECKPOINT = Path(__file__).parent / "checkpoints" / "names.json"
CONTEXT_LENGTH = 16
MAX_BODY_BYTES = 4096


class GenerationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [{"prefix": "ka", "count": 3, "seed": 42}]},
    )

    prefix: str = Field(
        default="",
        max_length=CONTEXT_LENGTH - 1,
        pattern=r"^[a-z]*$",
        description="Lowercase letters to keep at the start of each name.",
    )
    count: int = Field(default=1, ge=1, le=20)
    temperature: float = Field(
        default=0.8,
        ge=0,
        le=2,
        allow_inf_nan=False,
        description="Sampling temperature. Use 0 for the most likely continuation.",
    )
    seed: int = Field(
        default_factory=lambda: secrets.randbits(32),
        ge=0,
        le=2**32 - 1,
        description="Random seed. Omit it to choose a new seed for each request.",
    )
    max_new_tokens: int = Field(
        default=CONTEXT_LENGTH,
        ge=1,
        le=CONTEXT_LENGTH,
        description="Maximum new characters per name; total length is at most 16.",
    )


class GenerationResponse(BaseModel):
    model: Literal["tiny-names-gpt"] = "tiny-names-gpt"
    names: list[str]
    seed: int


class ModelInfo(BaseModel):
    model: Literal["tiny-names-gpt"] = "tiny-names-gpt"
    parameters: int
    context_length: int
    characters: str
    generation_endpoint: Literal["/generate"] = "/generate"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    detail: str


class GenerationBodyLimit:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in ("/generate", "/v1/responses")
        ):
            await self.app(scope, receive, send)
            return
        content_type = (
            Headers(scope=scope).get("content-type", "").split(";")[0].strip().lower()
        )
        responses = scope["path"] == "/v1/responses"
        limit = 65536 if responses else MAX_BODY_BYTES
        if content_type != "application/json":
            response = (
                APIError(
                    "Use Content-Type: application/json.",
                    code="invalid_content_type",
                    status=415,
                ).response()
                if responses
                else JSONResponse(
                    status_code=415,
                    content={"detail": "Use Content-Type: application/json."},
                )
            )
            await response(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit:
                response = (
                    APIError(
                        "Send a JSON body of at most 65536 bytes.",
                        code="request_too_large",
                        status=413,
                    ).response()
                    if responses
                    else JSONResponse(
                        status_code=413,
                        content={"detail": "Send a JSON body of at most 4096 bytes."},
                    )
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        pending = True

        async def receive_body():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, receive_body, send)


def create_app(checkpoint: Path = CHECKPOINT, *, response_settings=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load_model(checkpoint)
        app.state.capacity = threading.BoundedSemaphore(2)
        try:
            yield
        finally:
            app.state.responses.backend = None
            del app.state.model
            del app.state.capacity

    app = FastAPI(
        title="Language Model API",
        description="Generate text with a model trained from scratch using the Responses API.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(GenerationBodyLimit)
    install_routes(app, response_settings)

    @app.middleware("http")
    async def prevent_caching(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        if request.url.path.startswith("/v1/"):
            return response_validation_error(error)
        # Keep errors JSON-safe even when the supplied number is NaN or infinity.
        details = [
            {key: item[key] for key in ("loc", "msg", "type")}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        if request.url.path.startswith("/v1/"):
            return APIError(
                str(error.detail),
                code="not_found" if error.status_code == 404 else "invalid_request",
                status=error.status_code,
            ).response()
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": error.detail},
            headers=error.headers,
        )

    @app.get("/healthz", summary="Check service health")
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/", summary="Get model information")
    async def model_info(request: Request) -> ModelInfo:
        model = request.app.state.model
        return ModelInfo(
            parameters=len(model.parameters),
            context_length=model.config.context,
            characters=model.tokenizer.characters,
        )

    @app.post(
        "/generate",
        summary="Generate names",
        responses={
            413: {
                "model": ErrorResponse,
                "description": "Request body exceeds 4096 bytes.",
            },
            415: {
                "model": ErrorResponse,
                "description": "Content-Type must be application/json.",
            },
            429: {
                "model": ErrorResponse,
                "description": "Both generation slots are busy.",
            },
        },
    )
    def generate(payload: GenerationRequest, request: Request) -> GenerationResponse:
        capacity = request.app.state.capacity
        if not capacity.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Generation is busy. Try again shortly.",
                headers={"Retry-After": "1"},
            )
        try:
            names = [
                request.app.state.model.generate(
                    payload.prefix,
                    max_new_tokens=payload.max_new_tokens,
                    temperature=payload.temperature,
                    seed=payload.seed + index,
                )
                for index in range(payload.count)
            ]
            return GenerationResponse(names=names, seed=payload.seed)
        finally:
            capacity.release()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
