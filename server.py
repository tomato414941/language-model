"""Serve trained models through the Responses API and a chat page."""

import html
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from responses_api import (
    APIError,
    install_routes,
)
from responses_api import (
    validation_error as response_validation_error,
)

MAX_BODY_BYTES = 65536
CHAT_PAGE = Path(__file__).with_name("chat.html")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class GenerationBodyLimit:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/v1/responses"
        ):
            await self.app(scope, receive, send)
            return
        content_type = (
            Headers(scope=scope).get("content-type", "").split(";")[0].strip().lower()
        )
        if content_type != "application/json":
            response = APIError(
                "Use Content-Type: application/json.",
                code="invalid_content_type",
                status=415,
            ).response()
            await response(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
                response = APIError(
                    f"Send a JSON body of at most {MAX_BODY_BYTES} bytes.",
                    code="request_too_large",
                    status=413,
                ).response()
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


def create_app(*, response_settings=None, backend=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            app.state.responses.backend = None

    app = FastAPI(
        title="Language Model API",
        description="Generate text with a language model using the Responses API.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(GenerationBodyLimit)
    install_routes(app, response_settings, backend)
    settings = app.state.responses.settings
    chat_page = (
        CHAT_PAGE.read_text(encoding="utf-8")
        .replace("{{model}}", html.escape(settings.model_id))
        .replace("{{max_output_tokens}}", str(settings.max_output_tokens))
    )

    @app.middleware("http")
    async def prevent_caching(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return response_validation_error(error)

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

    @app.get("/", include_in_schema=False)
    async def chat() -> HTMLResponse:
        return HTMLResponse(chat_page)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
