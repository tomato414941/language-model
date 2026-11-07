"""The world outside the controller, on this machine, for the rental rehearsal.

FakeStore serves presigned URLs as R2 does in the respects the run depends on: a
URL is good for one object and one operation until it expires, one PUT takes a
limited size, ranges come back as asked, and parts of a multipart upload are
joined when the controller completes it. FakeProvider rents machines that are
directories on this machine, runs commands in them, returns them at their
deadline as vast-rental's watcher does, and lets a test kill or slow one.
"""

import hashlib
import hmac
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from rental.controller import Unreachable

# Every URL the fake store makes carries this, so a test can look for leaks.
SIGNATURE = "rehearsal-signature"


class FakeStore:
    def __init__(self, max_put_bytes=64 * 1024):
        self.max_put_bytes = max_put_bytes
        self.objects, self.uploads, self.refused = {}, {}, []
        self.joined = 0
        self.secret = os.urandom(16)
        self.lock = threading.Lock()
        self.on_object = None
        store = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                store.serve(self, "GET")

            def do_PUT(self):
                store.serve(self, "PUT")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    # URLs

    def sign(self, operation, key, upload, part, expires):
        message = "\n".join([operation, key, upload, str(part), str(expires)]).encode()
        return hmac.new(self.secret, message, hashlib.sha256).hexdigest()

    def presign(self, operation, key, seconds, upload="", part=0):
        expires = round(time.time() + seconds, 3)
        query = f"op={operation}&upload={upload}&part={part}&expires={expires}"
        return f"{self.base}/{quote(key)}?{query}&{SIGNATURE}={self.sign(operation, key, upload, part, expires)}"

    def presign_get(self, key, seconds):
        return self.presign("get", key, seconds)

    def presign_put(self, key, seconds):
        return self.presign("put", key, seconds)

    def presign_part(self, key, upload_id, number, seconds):
        return self.presign("part", key, seconds, upload_id, number)

    def serve(self, handler, method):
        address = urlsplit(handler.path)
        key = unquote(address.path.lstrip("/"))
        query = {name: values[0] for name, values in parse_qs(address.query).items()}
        operation, upload, part = query.get("op", ""), query.get("upload", ""), int(query.get("part", 0))
        expected = self.sign(operation, key, upload, part, query.get("expires", ""))
        if not hmac.compare_digest(expected, query.get(SIGNATURE, "")) or (method == "GET") != (operation == "get"):
            return self.refuse(handler, 403, key, "signature")
        if time.time() > float(query["expires"]):
            return self.refuse(handler, 403, key, "expired")
        if method == "GET":
            return self.send_object(handler, key)
        body = handler.rfile.read(int(handler.headers["Content-Length"]))
        if len(body) > self.max_put_bytes:
            return self.refuse(handler, 400, key, "too large")
        etag = hashlib.md5(body).hexdigest()
        with self.lock:
            if operation == "put":
                self.objects[key] = body
            elif upload in self.uploads:
                self.uploads[upload]["parts"][part] = body
            else:
                return self.refuse(handler, 404, key, "no such upload")
        handler.send_response(200)
        handler.send_header("ETag", f'"{etag}"')
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        if operation == "put" and self.on_object:
            self.on_object(key)

    def refuse(self, handler, code, key, reason):
        self.refused.append((key, reason))
        handler.send_response(code)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def send_object(self, handler, key):
        data = self.objects.get(key)
        if data is None:
            return self.refuse(handler, 404, key, "missing")
        requested = handler.headers.get("Range")
        if requested:
            start, end = (int(value) for value in requested.removeprefix("bytes=").split("-"))
            part = data[start:end + 1]
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes {start}-{start + len(part) - 1}/{len(data)}")
        else:
            part = data
            handler.send_response(200)
        handler.send_header("Content-Length", str(len(part)))
        handler.end_headers()
        handler.wfile.write(part)

    # What the controller does with its key

    def start_multipart(self, key):
        upload = uuid.uuid4().hex
        with self.lock:
            self.uploads[upload] = {"key": key, "parts": {}}
        return upload

    def complete_multipart(self, key, upload_id, parts):
        with self.lock:
            upload = self.uploads.pop(upload_id)
            if upload["key"] != key or sorted(upload["parts"]) != [number for number, _ in parts]:
                raise ValueError("the parts do not match the upload")
            for number, etag in parts:
                if hashlib.md5(upload["parts"][number]).hexdigest() != etag:
                    raise ValueError("a part's ETag does not match")
            self.objects[key] = b"".join(upload["parts"][number] for number, _ in parts)
            self.joined += 1
        if self.on_object:
            self.on_object(key)

    def abort_multipart(self, key, upload_id):
        with self.lock:
            self.uploads.pop(upload_id, None)

    def head(self, key):
        data = self.objects.get(key)
        return None if data is None else len(data)

    def read_json(self, key):
        data = self.objects.get(key)
        return None if data is None else json.loads(data)

    def write_json(self, key, value):
        self.objects[key] = json.dumps(value).encode()


class FakeProvider:
    """Machines that are directories here. Offers are dictionaries of price, speed and how many there are."""

    def __init__(self, directory, offers, max_price):
        self.directory = Path(directory)
        self.offers, self.max_price = offers, max_price
        self.machines_, self.notes, self.rented = {}, [], []
        self.lock = threading.Lock()

    def machines(self):
        with self.lock:
            for ident, machine in list(self.machines_.items()):
                if time.time() > machine["deadline"]:
                    self.stop_processes(machine)
                    machine["returned"] = "deadline"
            return [{"id": ident, "price": machine["price"]}
                    for ident, machine in self.machines_.items() if not machine.get("returned")]

    def rent(self, hours):
        with self.lock:
            fitting = sorted((offer for offer in self.offers if offer["price"] <= self.max_price and offer.get("count", 1) > 0),
                             key=lambda offer: offer["price"])
            if not fitting:
                return None
            offer = fitting[0]
            offer["count"] = offer.get("count", 1) - 1
            ident = f"m{len(self.rented) + 1}"
            directory = self.directory / ident
            directory.mkdir(parents=True)
            self.machines_[ident] = {"directory": directory, "price": offer["price"], "speed": offer["speed"],
                                     "deadline": time.time() + hours * 3600, "answering": True}
            self.rented.append(ident)
            return ident

    def extend(self, ident, hours):
        self.machines_[ident]["deadline"] = time.time() + hours * 3600

    def release(self, ident):
        machine = self.machines_[ident]
        self.stop_processes(machine)
        machine["returned"] = machine.get("returned") or "released"

    def note(self, ident, text):
        self.notes.append((ident, text))

    def work_dir(self, ident):
        return str(self.machines_[ident]["directory"])

    def run(self, ident, command, data=None, timeout=300):
        machine = self.machines_[ident]
        if not machine["answering"] or machine.get("returned"):
            raise Unreachable(ident)
        environment = {"PATH": os.environ["PATH"], "HOME": str(machine["directory"]),
                       "REHEARSAL_SPEED": str(machine["speed"]), "PYTHONDONTWRITEBYTECODE": "1"}
        completed = subprocess.run(["bash", "-c", command], input=data, capture_output=True, text=True,
                                   timeout=timeout, cwd=machine["directory"], env=environment)
        return completed.returncode, completed.stdout

    # What only a test does

    def kill(self, ident):
        """The machine dies: its processes end and it answers no more."""
        machine = self.machines_[ident]
        self.stop_processes(machine)
        machine["answering"] = False

    @staticmethod
    def stop_processes(machine):
        """Kill what runs in the machine's directory; a PID now used by a process elsewhere is left alone."""
        directory = machine["directory"].resolve()
        agent = directory / "lm/.agent"
        for path in [*agent.glob("pids/*.pid"), agent / "setup.pid"]:
            try:
                pid = int(path.read_text())
                if Path(os.readlink(f"/proc/{pid}/cwd")).resolve().is_relative_to(directory):
                    os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
                pass

    def return_all(self):
        for ident in list(self.machines_):
            self.release(ident)
