"""Serve the trained character model over HTTP."""

import json
import math
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model import load_model

CHECKPOINT = Path(__file__).parent / "checkpoints" / "names.json"


def generate(model, payload):
    if not isinstance(payload, dict):
        raise ValueError("Send a JSON object.")
    allowed = {"prefix", "count", "temperature", "seed", "max_new_tokens"}
    if payload.keys() - allowed:
        raise ValueError("Unknown generation option.")
    prefix = payload.get("prefix", "")
    count = payload.get("count", 1)
    temperature = payload.get("temperature", 0.8)
    seed = payload.get("seed", random.SystemRandom().randrange(2**32))
    limit = payload.get("max_new_tokens", model.config.context)
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string.")
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("count must be an integer between 1 and 20.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 4294967295.")
    if type(limit) is not int or not 1 <= limit <= model.config.context:
        raise ValueError(f"max_new_tokens must be between 1 and {model.config.context}.")
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2 or not math.isfinite(temperature):
        raise ValueError("temperature must be a finite number between 0 and 2.")
    names = [
        model.generate(prefix, max_new_tokens=limit, temperature=temperature, seed=seed + index)
        for index in range(count)
    ]
    return {"model": "tiny-names-gpt", "names": names, "seed": seed}


def create_server(host="127.0.0.1", port=0, checkpoint=CHECKPOINT):
    model = load_model(checkpoint)
    capacity = threading.BoundedSemaphore(2)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                self.reply(200, {"status": "ok"})
            elif self.path == "/":
                self.reply(200, {
                    "model": "tiny-names-gpt",
                    "parameters": len(model.parameters),
                    "context_length": model.config.context,
                    "characters": model.tokenizer.characters,
                    "generation_endpoint": "/generate",
                })
            else:
                self.reply(404, {"error": "Endpoint not found."})

        def do_POST(self):
            if self.path != "/generate":
                self.reply(404, {"error": "Endpoint not found."})
                return
            if self.headers.get_content_type() != "application/json":
                self.reply(415, {"error": "Use Content-Type: application/json."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Send a JSON body of at most 4096 bytes.")
                payload = json.loads(self.rfile.read(size))
            except (ValueError, UnicodeDecodeError):
                self.reply(400, {"error": "Send a valid JSON body of at most 4096 bytes."})
                return
            if not capacity.acquire(blocking=False):
                self.reply(429, {"error": "Generation is busy. Try again shortly."})
                return
            try:
                self.reply(200, generate(model, payload))
            except ValueError as error:
                self.reply(400, {"error": str(error)})
            finally:
                capacity.release()

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    server = create_server("0.0.0.0", int(os.environ.get("PORT", "8000")))
    print(f"Listening on port {server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
