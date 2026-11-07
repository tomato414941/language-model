"""Keep one healthy rented machine working on a run until its results are in storage, then return it.

The controller is the run's only authority. It rents through a provider, runs
commands on the machines through the provider, and alone holds the storage key:
a machine gets a presigned URL for each transfer when the transfer is due, on
standard input, valid only for that transfer. The progress record in storage
names the finished steps and the files kept, so any machine, and a controller
started again, takes the run up from there.

Each tick observes every machine, decides and acts:
- no machine at work: rent one when an offer fits, or wait;
- a new machine: send it the snapshot; once it is set up, the files to fetch and
  the steps to run;
- files in a machine's outbox: authorize their upload, then verify and record them;
- a machine that stops answering, whose run keeps stopping, or that trains much
  slower than expected: hand the run over and replace it;
- the run complete, failed or over its budget: return the machines and stop.

Machines are rented with a deadline at which the provider returns them, even if
no controller is left to; the controller moves the deadline on while it watches.
"""

import json
import math
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path


class Unreachable(Exception):
    """The machine did not answer; a command that ran and failed is not this."""


@dataclass
class Settings:
    tick_seconds: float = 60
    # Machines are rented for lease_hours and the lease is renewed when less than renew_hours is left.
    lease_hours: float = 6
    renew_hours: float = 3
    # How long one transfer's URLs stay valid.
    url_seconds: float = 6 * 3600
    unreachable_seconds: float = 15 * 60
    setup_seconds: float = 60 * 60
    # A runner that reports nothing for this long is stuck.
    stale_seconds: float = 15 * 60
    # A runner that stopped by itself is started again this many times on a machine.
    runner_restarts: int = 2
    # A step trains slow below slow_ratio of its expected rate for slow_seconds, once
    # warmup_seconds have passed since it started; the machine is replaced when the
    # time that saves exceeds replace_seconds, what a new machine costs to start.
    slow_ratio: float = 0.85
    slow_seconds: float = 30 * 60
    warmup_seconds: float = 30 * 60
    replace_seconds: float = 60 * 60
    # An upload not confirmed by then is authorized again.
    send_seconds: float = 3 * 3600
    send_attempts: int = 5
    # How long a machine being replaced has to save and send its checkpoint.
    handover_seconds: float = 30 * 60
    part_bytes: int = 1024**3
    budget_dollars: float | None = None


STATUS = """cd {work}/lm 2>/dev/null || {{ echo '{{"phase": "empty"}}'; exit 0; }}
if [ -f .agent/ready ]; then exec {python} -m rental.agent status; fi
if [ -f .agent/setup.exit ]; then echo '{{"phase": "setup_failed"}}'; exit 0; fi
if [ -f .agent/setup.pid ] && kill -0 "$(cat .agent/setup.pid)" 2>/dev/null; then
  echo '{{"phase": "setup"}}'
else
  echo '{{"phase": "empty"}}'
fi
"""
# The snapshot's URL comes on standard input. Extracting it again is skipped, so
# sending this again after an interruption picks up where it stopped.
BOOTSTRAP = """set -eu
mkdir -p {work}
cd {work}
read -r url
if [ ! -f lm/.agent/extracted ]; then
{prepare}
  rm -rf lm.tmp
  mkdir -p lm.tmp
  curl -fsSL --retry 5 "$url" | tar -xz -C lm.tmp
  rm -rf lm
  mv lm.tmp lm
  mkdir -p lm/.agent
  touch lm/.agent/extracted
fi
cd lm
cat > .agent/setup.sh <<'SETUP'
{setup}
SETUP
rm -f .agent/setup.exit
nohup bash -c 'bash .agent/setup.sh; code=$?; if [ $code = 0 ]; then touch .agent/ready; else echo $code > .agent/setup.exit; fi' \\
  > .agent/setup.log 2>&1 < /dev/null &
echo $! > .agent/setup.pid
"""
AGENT = "cd {work}/lm && exec {python} -m rental.agent {command}"
# Only these parts of the run's prefix take files from a machine.
KEPT_DIRECTORIES = ("checkpoints", "outputs", "logs")
URL = re.compile(r"https?://\S+")


def clean(value):
    """Anything URL-like out of what the controller writes down."""
    if isinstance(value, str):
        return URL.sub("<url>", value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


class Controller:
    def __init__(self, spec, provider, store, directory, settings=None, clock=time.time, sleep=time.sleep):
        self.spec, self.provider, self.store = spec, provider, store
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings = settings or Settings()
        self.clock, self.sleep = clock, sleep
        self.state = self.load(self.directory / "state.json") or {
            "machines": {}, "sending": {}, "outcome": None, "spent": 0.0, "setup_failures": 0, "waiting": False,
        }
        self.progress = self.store.read_json(self.key("progress.json")) or {
            "finished": [], "files": {}, "checkpoints": {}, "logs": {},
        }

    # Records

    @staticmethod
    def load(path):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None

    def save(self):
        temporary = self.directory / "state.json.tmp"
        temporary.write_text(json.dumps(self.state, indent=1) + "\n")
        temporary.replace(self.directory / "state.json")

    def event(self, name, **details):
        with (self.directory / "events.jsonl").open("a") as log:
            log.write(json.dumps(clean({"at": round(self.clock(), 3), "event": name, **details})) + "\n")

    def key(self, relative):
        return f"{self.spec['prefix']}/{relative}"

    def kept_key(self, relative):
        """The full key of a file a machine sends, which must stay in the run's own directories."""
        parts = relative.split("/")
        if parts[0] not in KEPT_DIRECTORIES or len(parts) < 2 or any(part in ("", ".", "..") for part in parts):
            raise ValueError("a machine asked to keep a file outside the run's directories")
        return self.key(relative)

    def record_progress(self):
        self.store.write_json(self.key("progress.json"), self.progress)

    # Commands on a machine

    def command(self, ident, template, data=None, timeout=300, **fields):
        work = shlex.quote(self.provider.work_dir(ident))
        text = template.format(work=work, python=self.spec["python"], **fields)
        return self.provider.run(ident, text, data=data, timeout=timeout)

    def agent(self, ident, command, given=None):
        payload = None if given is None else json.dumps(given)
        code, output = self.command(ident, AGENT, data=payload, command=command)
        if code:
            raise RuntimeError(f"the agent's {command} ended with {code}")
        return json.loads(output.strip().splitlines()[-1])

    def status(self, ident):
        code, output = self.command(ident, STATUS)
        if code:
            raise RuntimeError(f"the status ended with {code}")
        return json.loads(output.strip().splitlines()[-1])

    def bootstrap(self, ident):
        url = self.store.presign_get(self.key(self.spec["snapshot"]), self.settings.url_seconds)
        prepare = "\n".join(f"  {line}" for line in self.spec.get("prepare", "true").splitlines())
        code, _ = self.command(ident, BOOTSTRAP, data=url + "\n", timeout=1800,
                               prepare=prepare, setup=self.spec.get("setup", "true"))
        machine = self.state["machines"][ident]
        self.event("bootstrapped", machine=ident, exit_code=code)
        if code == 0:
            machine["bootstrapped_at"] = self.clock()
            return
        # Sent again at the next tick; a machine that keeps refusing it goes.
        machine["bootstrap_failures"] = machine.get("bootstrap_failures", 0) + 1
        if machine["bootstrap_failures"] >= 3:
            self.replace(ident, "the snapshot did not arrive", graceful=False)

    def fetch_list(self):
        """What a machine fetches before its steps: the data, the finished steps' files and the step's checkpoint."""
        seconds = self.settings.url_seconds
        files = [
            {"path": item["path"], "url": self.store.presign_get(item["key"], seconds),
             "sha256": item["sha256"], "bytes": item["bytes"]}
            for item in self.spec["data"]
        ]
        files += [
            {"path": source, "url": self.store.presign_get(self.key(item["key"]), seconds),
             "sha256": item["sha256"], "bytes": item["bytes"]}
            for source, item in sorted(self.progress["files"].items())
        ]
        current = self.current_step()
        checkpoint = current and self.progress["checkpoints"].get(current["name"])
        if checkpoint:
            files.append({"path": checkpoint["source"], "url": self.store.presign_get(self.key(checkpoint["key"]), seconds),
                          "sha256": checkpoint["sha256"], "bytes": checkpoint["bytes"], "if_missing": True})
        return files

    def start_runner(self, ident, redo=()):
        machine = self.state["machines"][ident]
        given = {
            "spec": {key: self.spec[key] for key in ("steps", "sync_seconds", "heartbeat_seconds", "rate_seconds")
                     if key in self.spec},
            "fetch": self.fetch_list(),
            "finished": self.progress["finished"],
            "redo": list(redo),
        }
        answer = self.agent(ident, "run", given)
        machine["runner_started_at"] = self.clock()
        self.event("runner_started", machine=ident, started=answer.get("started"),
                   resumed=bool(self.progress["finished"] or self.progress["checkpoints"]))

    # The steps

    def current_step(self):
        return next((step for step in self.spec["steps"] if step["name"] not in self.progress["finished"]), None)

    def note_finished(self, runner, entries):
        """A step the runner did is finished once nothing of it waits in the outbox and its outputs are kept."""
        pending = {entry["step"] for entry in entries}
        for step in self.spec["steps"]:
            name = step["name"]
            if name in self.progress["finished"] or name not in runner.get("done_steps", []) or name in pending:
                continue
            kept = self.progress["files"]
            if all(any(source == path or source.startswith(f"{path}/") for source in kept)
                   for path in step.get("outputs", [])):
                self.progress["finished"].append(name)
                self.progress["checkpoints"].pop(name, None)
                self.record_progress()
                self.event("step_finished", step=name)

    # Uploads

    def authorize(self, ident, entry):
        key = self.kept_key(entry["key"])
        seconds = self.settings.url_seconds
        earlier = self.state["sending"].get(entry["id"])
        if earlier and earlier.get("upload_id"):
            self.store.abort_multipart(earlier["key"], earlier["upload_id"])
        record = {"machine": ident, "key": key, "authorized_at": self.clock(),
                  "attempts": (earlier or {}).get("attempts", 0) + 1}
        if entry["bytes"] <= self.store.max_put_bytes:
            upload = {"id": entry["id"], "url": self.store.presign_put(key, seconds)}
        else:
            size = min(self.settings.part_bytes, self.store.max_put_bytes)
            size = max(size, math.ceil(entry["bytes"] / 10_000))
            record["upload_id"] = self.store.start_multipart(key)
            upload = {"id": entry["id"], "parts": [
                {"number": number, "start": start, "length": min(size, entry["bytes"] - start),
                 "url": self.store.presign_part(key, record["upload_id"], number, seconds)}
                for number, start in enumerate(range(0, entry["bytes"], size), 1)
            ]}
        self.state["sending"][entry["id"]] = record
        return upload

    def keep(self, entry):
        """Complete and check an upload the machine reports sent, and record the file as kept."""
        record = self.state["sending"][entry["id"]]
        if record.get("upload_id"):
            self.store.complete_multipart(record["key"], record["upload_id"], entry["etags"])
        if self.store.head(record["key"]) != entry["bytes"]:
            raise ValueError("the store holds another size")
        item = {"key": record["key"].removeprefix(self.key("")), "sha256": entry["sha256"], "bytes": entry["bytes"]}
        if entry["kind"] == "checkpoint":
            stored = self.progress["checkpoints"].get(entry["step"])
            if not stored or stored["registered_at"] <= entry["registered_at"]:
                self.progress["checkpoints"][entry["step"]] = {
                    **item, "source": entry["source"], "registered_at": entry["registered_at"]}
        else:
            self.progress["files" if entry["kind"] == "output" else "logs"][entry["source"]] = item
        self.record_progress()
        del self.state["sending"][entry["id"]]

    def tend_outbox(self, ident, entries):
        now, uploads, kept = self.clock(), [], []
        for entry in entries:
            record = self.state["sending"].get(entry["id"])
            if entry["state"] == "sent" and record:
                try:
                    self.keep(entry)
                    kept.append(entry["id"])
                    self.event("kept", machine=ident, kind=entry["kind"], step=entry["step"], bytes=entry["bytes"])
                except (ValueError, KeyError) as error:
                    self.event("keep_failed", machine=ident, step=entry["step"], error=type(error).__name__)
                    uploads.append(self.authorize(ident, entry))
                continue
            again = (
                (entry["state"] == "ready" and not record)
                or entry["state"] == "failed"
                or (entry["state"] in ("sending", "sent") and not record)
                or (entry["state"] == "sending" and now - record["authorized_at"] > self.settings.send_seconds)
            )
            if again:
                if record and record["attempts"] >= self.settings.send_attempts:
                    return self.end("failed", reason="an upload keeps failing", step=entry["step"])
                uploads.append(self.authorize(ident, entry))
        if uploads:
            self.agent(ident, "send", {"uploads": uploads})
            self.event("authorized", machine=ident, files=len(uploads))
        if kept:
            self.agent(ident, "ack", {"kept": kept})
        # Uploads the machine no longer lists will not be completed.
        listed = {entry["id"] for entry in entries}
        for ident_, record in list(self.state["sending"].items()):
            if record["machine"] == ident and ident_ not in listed:
                if record.get("upload_id"):
                    self.store.abort_multipart(record["key"], record["upload_id"])
                del self.state["sending"][ident_]

    # Machines

    def release(self, ident, reason):
        machine = self.state["machines"].pop(ident, None)
        self.provider.release(ident)
        if machine:
            self.state["spent"] += machine["price"] * (self.clock() - machine["rented_at"]) / 3600
        for key, record in list(self.state["sending"].items()):
            if record["machine"] == ident:
                if record.get("upload_id"):
                    self.store.abort_multipart(record["key"], record["upload_id"])
                del self.state["sending"][key]
        self.event("released", machine=ident, reason=reason)

    def replace(self, ident, reason, graceful):
        """Note why the machine goes; let it save and send its checkpoint first when it can."""
        self.provider.note(ident, reason)
        self.event("replacing", machine=ident, reason=reason)
        if graceful:
            try:
                self.agent(ident, "stop")
                self.state["machines"][ident]["handover"] = {"reason": reason, "since": self.clock()}
                return
            except (Unreachable, RuntimeError, ValueError):
                pass
        self.release(ident, reason)

    def slow(self, ident, runner):
        """Whether to replace a machine whose step trains well below its expected rate."""
        machine = self.state["machines"][ident]
        step = next((item for item in self.spec["steps"] if item["name"] == runner.get("step")), None)
        rate, now = runner.get("rate"), self.clock()
        expected = step and step.get("expected_rate")
        if not expected or rate is None or now - runner.get("step_started", now) < self.settings.warmup_seconds:
            machine["slow_since"] = None
            return False
        if rate >= self.settings.slow_ratio * expected:
            machine["slow_since"] = None
            return False
        machine["slow_since"] = machine.get("slow_since") or now
        if now - machine["slow_since"] < self.settings.slow_seconds:
            return False
        # The time a machine at the expected rate would save on what is left of the trainings.
        left = 0.0
        for item in self.spec["steps"][self.spec["steps"].index(step):]:
            if item.get("expected_rate") and item.get("units"):
                done = (runner.get("progress") or 0) if item is step else 0
                left += max(0, item["units"] - done) / item["expected_rate"]
        saved = left * (expected / max(rate, 1e-9) - 1)
        return saved > self.settings.replace_seconds

    def look_after(self, ident, row):
        now = self.clock()
        machine = self.state["machines"].setdefault(ident, {
            "rented_at": now, "price": row.get("price", 0.0), "deadline": now + self.settings.lease_hours * 3600,
            "restarts": 0,
        })
        try:
            report = self.status(ident)
        except (Unreachable, RuntimeError, ValueError) as error:
            machine["unreachable_since"] = machine.get("unreachable_since") or now
            if now - machine["unreachable_since"] >= self.settings.unreachable_seconds:
                self.replace(ident, f"no answer: {type(error).__name__}", graceful=False)
            return
        machine["unreachable_since"] = None
        try:
            self.attend(ident, machine, report)
        except (Unreachable, RuntimeError, ValueError) as error:
            # A command that did not go through is tried again at the next tick.
            self.event("command_failed", machine=ident, error=type(error).__name__)

    def attend(self, ident, machine, report):
        now = self.clock()
        phase = report["phase"]
        if phase == "empty":
            if now - machine.get("bootstrapped_at", 0) > self.settings.setup_seconds:
                self.bootstrap(ident)
            return
        if phase == "setup":
            if now - machine.get("bootstrapped_at", now) > self.settings.setup_seconds:
                self.replace(ident, "setup took too long", graceful=False)
            return
        if phase == "setup_failed":
            self.state["setup_failures"] += 1
            self.replace(ident, "setup failed", graceful=False)
            if self.state["setup_failures"] >= 2:
                self.end("failed", reason="setup failed on two machines")
            return
        self.tend_outbox(ident, report["outbox"])
        if self.state["outcome"] or ident not in self.state["machines"]:
            return
        runner = report["runner"]
        self.note_finished(runner, [entry for entry in report["outbox"] if entry["state"] != "superseded"])
        waiting = report["outbox"]
        if machine.get("handover"):
            if (not runner["alive"] and not waiting) or now - machine["handover"]["since"] > self.settings.handover_seconds:
                self.release(ident, machine["handover"]["reason"])
            return
        self.renew(ident, machine)
        state = runner.get("state", "idle")
        if runner["alive"]:
            if now - runner.get("heartbeat", now) > self.settings.stale_seconds:
                # Stopped once; a runner that does not stop either leaves with its machine.
                if machine.get("stopping_since") is None:
                    machine["stopping_since"] = now
                    self.event("stopping", machine=ident, reason="the runner stopped reporting")
                    self.agent(ident, "stop")
                elif now - machine["stopping_since"] > self.settings.stale_seconds:
                    self.replace(ident, "the runner is stuck", graceful=False)
            elif self.slow(ident, runner):
                self.replace(ident, f"slow: {runner['rate']:.0f} a second on {runner['step']}", graceful=True)
            return
        machine["stopping_since"] = None
        if state == "idle":
            self.start_runner(ident)
        elif state == "failed":
            if not waiting:
                self.end("failed", reason="a step failed", **runner.get("failure", {}))
        elif state == "done":
            # Steps the machine did whose files the progress record lacks, as after a lost
            # controller state, are done again.
            missing = [name for name in runner.get("done_steps", []) if name not in self.progress["finished"]]
            if not waiting and missing:
                self.event("redoing", machine=ident, steps=missing)
                self.start_runner(ident, redo=missing)
        else:
            self.restart(ident, machine, f"the runner stopped: {state}")

    def restart(self, ident, machine, reason):
        if machine["restarts"] >= self.settings.runner_restarts:
            self.replace(ident, f"{reason}, again", graceful=False)
            return
        machine["restarts"] += 1
        self.event("restarting", machine=ident, reason=reason)
        self.start_runner(ident)

    def renew(self, ident, machine):
        now = self.clock()
        if machine["deadline"] - now < self.settings.renew_hours * 3600:
            self.provider.extend(ident, self.settings.lease_hours)
            machine["deadline"] = now + self.settings.lease_hours * 3600
            self.event("renewed", machine=ident)

    def spent(self):
        now = self.clock()
        return self.state["spent"] + sum(
            machine["price"] * (now - machine["rented_at"]) / 3600 for machine in self.state["machines"].values())

    def end(self, outcome, **details):
        for ident in list(self.state["machines"]):
            self.release(ident, outcome)
        self.state["outcome"] = outcome
        self.event("ended", outcome=outcome, spent=round(self.spent(), 4), **details)
        self.save()
        return outcome

    def tick(self):
        if self.state["outcome"]:
            return self.state["outcome"]
        listed = {row["id"]: row for row in self.provider.machines()}
        for ident in list(self.state["machines"]):
            if ident not in listed:
                machine = self.state["machines"].pop(ident)
                self.state["spent"] += machine["price"] * (self.clock() - machine["rented_at"]) / 3600
                self.event("gone", machine=ident)
        for ident, row in listed.items():
            self.look_after(ident, row)
            if self.state["outcome"]:
                return self.state["outcome"]
        if self.current_step() is None:
            return self.end("complete")
        budget = self.settings.budget_dollars
        if budget is not None and self.spent() >= budget:
            return self.end("over budget")
        if not self.state["machines"]:
            ident = self.provider.rent(self.settings.lease_hours)
            if ident is None:
                if not self.state["waiting"]:
                    self.event("waiting", reason="no offer fits")
                self.state["waiting"] = True
            else:
                self.state["waiting"] = False
                now = self.clock()
                price = next((row.get("price", 0.0) for row in self.provider.machines() if row["id"] == ident), 0.0)
                self.state["machines"][ident] = {"rented_at": now, "price": price, "restarts": 0,
                                                 "deadline": now + self.settings.lease_hours * 3600}
                self.event("rented", machine=ident, price=price)
        self.save()
        return None

    def run(self, stop=None):
        """Tick until the run ends, or until stop() says to leave it as it is."""
        while True:
            outcome = self.tick()
            if outcome or (stop and stop()):
                return outcome
            self.sleep(self.settings.tick_seconds)
