"""The rented machine's side of a run: its steps, and the transfers the controller allows.

The machine holds no credential. The controller runs these commands on it from
the snapshot's root, over SSH or, in the rehearsal, a local shell, and passes
every URL on standard input. Work that outlasts the connection carries on in a
detached process that keeps its URLs in memory only.

  status  print the run's state and the files waiting to go to storage
  run     fetch the files given, then run the steps the controller has not seen finish
  send    upload files from the outbox to the URLs given
  ack     forget files the controller has kept
  stop    stop the run's processes

Files the run keeps go into the outbox as hard links, so that training may
replace the originals while a copy is on its way. A step's checkpoint goes in
every `sync_seconds`, and its outputs and log when it ends.
"""

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.cloud_files import Slice, transfer_file, upload_digests

ROOT = Path.cwd()
AGENT = ROOT / ".agent"
OUT = AGENT / "out"
LOGS = AGENT / "logs"
PIDS = AGENT / "pids"
DONE = AGENT / "done"
UPLOAD_ATTEMPTS = 3


def write_json(path, value):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(temporary, path)


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


@contextlib.contextmanager
def outbox():
    """The outbox's entries, read and written under a lock the runner and the senders share."""
    AGENT.mkdir(exist_ok=True)
    with (AGENT / "outbox.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        box = read_json(AGENT / "outbox.json", {"entries": []})
        yield box
        write_json(AGENT / "outbox.json", box)


def alive(name):
    """Whether the process the name's PID file records still runs this agent."""
    try:
        pid = int((PIDS / f"{name}.pid").read_text())
        os.kill(pid, 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    # A PID the system gave to another process since does not count.
    command = Path(f"/proc/{pid}/cmdline")
    return not command.exists() or b"rental.agent" in command.read_bytes()


def remember(name, pid):
    PIDS.mkdir(parents=True, exist_ok=True)
    (PIDS / f"{name}.pid").write_text(f"{pid}\n")


def detached(log):
    """Fork a process that outlives the connection: True in it, False in the caller."""
    child = os.fork()
    if child:
        os.waitpid(child, 0)
        return False
    os.setsid()
    if os.fork():
        os._exit(0)
    descriptor = os.open(os.devnull, os.O_RDONLY)
    os.dup2(descriptor, 0)
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    return True


def register(kind, step, source, key):
    """Put a file, or each file of a directory, in the outbox under a key below the run's prefix."""
    path = ROOT / source
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for item in files:
        relative = str(item.relative_to(ROOT))
        ident = uuid.uuid4().hex[:16]
        link = OUT / ident
        os.link(item, link)
        sha256, md5 = upload_digests(link)
        entries.append({
            "id": ident, "kind": kind, "step": step, "source": relative,
            "key": key if item == path else f"{key}/{item.relative_to(path)}",
            "sha256": sha256, "md5": md5, "bytes": link.stat().st_size,
            "state": "ready", "registered_at": time.time(),
        })
    with outbox() as box:
        if kind == "checkpoint":
            # Only the newest checkpoint of a step matters; an older one not yet on its way goes.
            for entry in box["entries"]:
                if entry["kind"] == "checkpoint" and entry["step"] == step and entry["state"] == "ready":
                    entry["state"] = "superseded"
                    (OUT / entry["id"]).unlink(missing_ok=True)
            box["entries"] = [entry for entry in box["entries"] if entry["state"] != "superseded"]
        box["entries"] += entries


class Runner:
    """Fetch the files given, then run each step the controller has not seen finish."""

    def __init__(self, spec, finished, redo=()):
        self.spec, self.finished = spec, set(finished)
        for name in redo:
            (DONE / name).unlink(missing_ok=True)
        self.status = read_json(AGENT / "status.json", {})
        self.status.update(state="starting", step=None, failure=None)
        self.status["done_steps"] = sorted(self.finished | {path.name for path in DONE.glob("*")})
        self.child, self.stopping = None, False

    def save(self, **changes):
        self.status.update(changes, heartbeat=time.time())
        write_json(AGENT / "status.json", self.status)

    def stop(self, *_):
        self.stopping = True
        if self.child and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)

    def fetch(self, files):
        self.save(state="fetching", fetched=0, fetching=len(files))
        for count, entry in enumerate(files, 1):
            # A checkpoint this machine saved itself is newer than the stored one.
            if not (entry.get("if_missing") and (ROOT / entry["path"]).exists()):
                transfer_file("download", entry, ROOT)
            self.save(fetched=count)

    def command(self, step):
        command = [sys.executable if part == "python" else part for part in step["command"]]
        checkpoint = step.get("checkpoint")
        if checkpoint and (ROOT / checkpoint).exists() and "resume" in step:
            dropped = set(step["resume"].get("drop", []))
            kept, skip = [], False
            for part in command:
                if skip:
                    skip = False
                elif part in dropped:
                    skip = True
                else:
                    kept.append(part)
            command = kept + step["resume"]["add"]
        return command

    def execute(self, step):
        """Run the step's command, reporting its rate and putting its checkpoint in the outbox now and then."""
        LOGS.mkdir(parents=True, exist_ok=True)
        heartbeat = self.spec.get("heartbeat_seconds", 30)
        window = self.spec.get("rate_seconds", 600)
        sync = self.spec.get("sync_seconds", 4 * 3600)
        metrics = ROOT / step["metrics"] if step.get("metrics") else None
        checkpoint = ROOT / step["checkpoint"] if step.get("checkpoint") else None
        observations, offset, synced, seen = deque(), 0, time.monotonic(), None
        self.save(state="running", step=step["name"], step_started=time.time(), rate=None,
                  progress=None, progressed_at=time.time())
        with (LOGS / f"{step['name']}.log").open("a") as log:
            self.child = subprocess.Popen(
                self.command(step), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, **step.get("environment", {})},
            )
            remember("child", self.child.pid)
            while self.child.poll() is None:
                time.sleep(min(1.0, heartbeat))
                now = time.time()
                if metrics and metrics.exists():
                    if metrics.stat().st_size < offset:
                        offset = 0
                    with metrics.open() as source:
                        source.seek(offset)
                        for line in source:
                            if not line.endswith("\n"):
                                break
                            offset += len(line.encode())
                            value = json.loads(line).get(step.get("metrics_key", "tokens_seen"))
                            if value is not None and (not observations or value > observations[-1][1]):
                                observations.append((now, value))
                                self.status.update(progress=value, progressed_at=now)
                while len(observations) > 1 and observations[1][0] < now - window:
                    observations.popleft()
                if observations and observations[-1][0] < now - window:
                    rate = 0.0
                elif len(observations) > 1 and observations[-1][0] - observations[0][0] >= window / 2:
                    (first, start), (last, end) = observations[0], observations[-1]
                    rate = (end - start) / (last - first)
                else:
                    rate = None
                if now - self.status["heartbeat"] >= heartbeat:
                    self.save(rate=rate)
                if checkpoint and time.monotonic() - synced >= sync and checkpoint.exists():
                    stat = checkpoint.stat()
                    if (stat.st_mtime_ns, stat.st_size) != seen:
                        seen, synced = (stat.st_mtime_ns, stat.st_size), time.monotonic()
                        register("checkpoint", step["name"], step["checkpoint"],
                                 f"checkpoints/{step['name']}/{checkpoint.name}")
        return self.child.returncode

    def keep_partial(self, step):
        """Put the step's checkpoint and log in the outbox, so a failed or stopped step loses nothing."""
        if step.get("checkpoint") and (ROOT / step["checkpoint"]).exists():
            register("checkpoint", step["name"], step["checkpoint"],
                     f"checkpoints/{step['name']}/{Path(step['checkpoint']).name}")
        if (LOGS / f"{step['name']}.log").exists():
            register("log", step["name"], f".agent/logs/{step['name']}.log", f"logs/{step['name']}.log")

    def run(self, files):
        signal.signal(signal.SIGTERM, self.stop)
        try:
            self.fetch(files)
        except (OSError, ValueError, HTTPException, KeyError) as error:
            self.save(state="fetch_failed", failure={"error": type(error).__name__})
            return
        DONE.mkdir(parents=True, exist_ok=True)
        for step in self.spec["steps"]:
            if step["name"] in self.status["done_steps"]:
                continue
            if self.stopping:
                break
            code = self.execute(step)
            if self.stopping:
                self.keep_partial(step)
                self.save(state="stopped")
                return
            if code != 0:
                self.keep_partial(step)
                self.save(state="failed", failure={"step": step["name"], "exit_code": code})
                return
            for source in step.get("outputs", []):
                register("output", step["name"], source, f"outputs/{source}")
            register("log", step["name"], f".agent/logs/{step['name']}.log", f"logs/{step['name']}.log")
            (DONE / step["name"]).touch()
            self.save(done_steps=sorted({*self.status["done_steps"], step["name"]}))
        self.save(state="stopped" if self.stopping else "done", step=None)


def finish(work):
    """Do the detached process's work and end it there, its traceback in its log if it breaks."""
    code = 0
    try:
        work()
    except BaseException as error:  # A detached process has no caller to raise to.
        traceback.print_exc()
        code = 1
        with contextlib.suppress(OSError):
            status = read_json(AGENT / "status.json", {})
            if status.get("state") in ("starting", "fetching", "running"):
                write_json(AGENT / "status.json", {**status, "state": "crashed",
                                                   "failure": {"error": type(error).__name__}})
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def put(path, url, start, length):
    """Send length bytes of the file from start; return the store's ETag. A refusal is not retried."""
    for attempt in range(UPLOAD_ATTEMPTS):
        try:
            with path.open("rb") as source:
                request = Request(url, data=Slice(source, start, length), method="PUT",
                                  headers={"Content-Length": str(length)})
                with urlopen(request, timeout=120) as response:
                    response.read()
                    return (response.headers.get("ETag") or "").strip('"')
        except HTTPError as error:
            if error.code < 500 or attempt + 1 == UPLOAD_ATTEMPTS:
                raise
        except (OSError, HTTPException):
            if attempt + 1 == UPLOAD_ATTEMPTS:
                raise
        time.sleep(2**attempt)


def send(uploads):
    """Upload each file to its URL or its parts' URLs, and record what the store confirmed."""
    for upload in uploads:
        with outbox() as box:
            entry = next((item for item in box["entries"] if item["id"] == upload["id"]), None)
        if entry is None:
            continue
        link = OUT / entry["id"]
        try:
            if "url" in upload:
                etag = put(link, upload["url"], 0, entry["bytes"])
                if etag != entry["md5"]:
                    raise ValueError("the store received other bytes")
                etags = [[0, etag]]
            else:
                etags = []
                for part in upload["parts"]:
                    expected = upload_digests(link, part["start"], part["length"])[1]
                    etag = put(link, part["url"], part["start"], part["length"])
                    if etag != expected:
                        raise ValueError("the store received other bytes")
                    etags.append([part["number"], etag])
            changes = {"state": "sent", "etags": etags}
        except (OSError, ValueError, HTTPException) as error:
            changes = {"state": "failed", "error": type(error).__name__}
        with outbox() as box:
            for item in box["entries"]:
                if item["id"] == upload["id"] and item["state"] == "sending":
                    item.update(changes)


def gpu():
    fields = ["name", "memory.used", "memory.total", "utilization.gpu", "power.draw", "power.limit",
              "clocks.sm", "clocks.max.sm"]
    try:
        listed = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listed.returncode:
        return None
    return [dict(zip(fields, [value.strip() for value in line.split(",")])) for line in listed.stdout.splitlines()]


def status(_args):
    # The state first: the runner puts a step's files in the outbox before it records the step as
    # done, so a step reported done has its files in the outbox read after.
    runner = {"alive": alive("runner"), **read_json(AGENT / "status.json", {"state": "idle"})}
    with outbox() as box:
        entries = box["entries"]
    report = {
        "phase": "ready",
        "runner": runner,
        "outbox": entries,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "gpu": gpu(),
    }
    print(json.dumps(report))


def run(_args):
    if alive("runner"):
        print(json.dumps({"started": False, "reason": "running"}))
        return
    given = json.load(sys.stdin)
    write_json(AGENT / "spec.json", given["spec"])
    LOGS.mkdir(parents=True, exist_ok=True)
    if not detached(LOGS / "runner.log"):
        print(json.dumps({"started": True}))
        return
    remember("runner", os.getpid())
    finish(lambda: Runner(given["spec"], given.get("finished", []), given.get("redo", [])).run(given.get("fetch", [])))


def send_command(_args):
    uploads = json.load(sys.stdin)["uploads"]
    wanted = {upload["id"] for upload in uploads}
    with outbox() as box:
        for entry in box["entries"]:
            if entry["id"] in wanted and entry["state"] in ("ready", "failed", "sending"):
                entry.update(state="sending", error=None)
    LOGS.mkdir(parents=True, exist_ok=True)
    if not detached(LOGS / "sender.log"):
        print(json.dumps({"sending": sorted(wanted)}))
        return
    name = f"sender-{os.getpid()}"
    remember(name, os.getpid())

    def work():
        try:
            send(uploads)
        finally:
            (PIDS / f"{name}.pid").unlink(missing_ok=True)

    finish(work)


def ack(_args):
    kept = set(json.load(sys.stdin)["kept"])
    with outbox() as box:
        for entry in box["entries"]:
            if entry["id"] in kept:
                (OUT / entry["id"]).unlink(missing_ok=True)
        box["entries"] = [entry for entry in box["entries"] if entry["id"] not in kept]
    print(json.dumps({"forgotten": sorted(kept)}))


def stop(_args):
    stopped = []
    for path in sorted(PIDS.glob("*.pid")) if PIDS.exists() else []:
        if path.stem == "child":
            continue
        with contextlib.suppress(ValueError, ProcessLookupError, PermissionError):
            os.kill(int(path.read_text()), signal.SIGTERM)
            stopped.append(path.stem)
    print(json.dumps({"stopping": stopped}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    for name, function in (("status", status), ("run", run), ("send", send_command), ("ack", ack), ("stop", stop)):
        commands.add_parser(name).set_defaults(function=function)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
