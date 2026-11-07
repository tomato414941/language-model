"""The rental rehearsal: a whole run's life on this machine, against a fake provider and store.

Only the outside world is fake: the provider's machines are directories here,
the store answers presigned URLs over HTTP, and training is a stand-in that
steps at its machine's speed. The controller, the agent and the transfers are
the ones a real run uses.
"""

import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rental.controller import Controller, Settings
from tests.rental_fakes import SIGNATURE, FakeProvider, FakeStore

PROJECT = Path(__file__).resolve().parents[1]
PREFIX = "stages/rehearsal/run"
FAST = dict(
    tick_seconds=0.05, lease_hours=60 / 3600, renew_hours=30 / 3600, url_seconds=30, unreachable_seconds=0.6,
    setup_seconds=10, stale_seconds=3, runner_restarts=2, slow_ratio=0.85, slow_seconds=0.4, warmup_seconds=0.3,
    replace_seconds=0.05, send_seconds=5, send_attempts=5, handover_seconds=5, part_bytes=32 * 1024,
)
EXPECTED_RATE = 1000 / 0.02


def train_step(name, steps, *extra, resume_drop=()):
    output = f"out/{name}"
    return {
        "name": name, "kind": "train",
        "command": ["python", "tests/rental_job.py", "train", "--output", output, "--steps", str(steps),
                    "--checkpoint-every", "10", *extra],
        "checkpoint": f"{output}/last.pt", "metrics": f"{output}/metrics.jsonl",
        "resume": {"drop": list(resume_drop), "add": ["--resume", f"{output}/last.pt"]},
        "outputs": [f"{output}/last.pt", f"{output}/metrics.jsonl"],
        "expected_rate": EXPECTED_RATE, "units": steps * 1000,
    }


class RentalRehearsalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = FakeStore(max_put_bytes=64 * 1024)
        self.provider = FakeProvider(self.root / "machines", [{"price": 0.5, "speed": 1.0, "count": 5}], max_price=0.6)
        corpus = os.urandom(50 * 1024)
        self.store.objects["data/pretrain/corpus.bin"] = corpus
        self.store.objects[f"{PREFIX}/snapshot.tar.gz"] = self.snapshot()
        self.spec = {
            "prefix": PREFIX, "snapshot": "snapshot.tar.gz", "python": shlex.quote(sys.executable),
            "prepare": "true", "setup": "true",
            "data": [{"path": "data/corpus.bin", "key": "data/pretrain/corpus.bin",
                      "sha256": hashlib.sha256(corpus).hexdigest(), "bytes": len(corpus)}],
            "steps": [
                train_step("first", 60, "--data", "data/corpus.bin"),
                train_step("second", 40, "--initialize-from", "out/first/last.pt", resume_drop=["--initialize-from"]),
                {"name": "evaluate", "kind": "command",
                 "command": ["python", "tests/rental_job.py", "evaluate", "--checkpoint", "out/second/last.pt",
                             "--output", "out/evaluation.json"],
                 "outputs": ["out/evaluation.json"]},
            ],
            "sync_seconds": 0.3, "heartbeat_seconds": 0.1, "rate_seconds": 0.4,
        }

    def tearDown(self):
        self.provider.return_all()
        self.store.close()
        self.directory.cleanup()

    @staticmethod
    def snapshot():
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            for name in ("rental/__init__.py", "rental/agent.py", "scripts/cloud_files.py", "tests/rental_job.py"):
                bundle.add(PROJECT / name, arcname=name)
        return archive.getvalue()

    def controller(self, **settings):
        return Controller(self.spec, self.provider, self.store, self.root / "control", Settings(**{**FAST, **settings}))

    def run_to_end(self, controller, seconds=60):
        ended = time.monotonic() + seconds
        outcome = controller.run(stop=lambda: time.monotonic() > ended)
        self.assertIsNotNone(outcome, "the run did not end in time")
        return outcome

    def events(self):
        path = self.root / "control/events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def machine_file(self, ident, relative):
        return self.root / "machines" / ident / "lm" / relative

    def assert_no_url_written(self):
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".sh"):
                self.assertNotIn(SIGNATURE, path.read_text(errors="replace"), path.name)

    def assert_results_kept(self):
        evaluation = json.loads(self.store.objects[f"{PREFIX}/outputs/out/evaluation.json"])
        self.assertEqual(evaluation, {"checkpoint_step": 40})
        progress = json.loads(self.store.objects[f"{PREFIX}/progress.json"])
        self.assertEqual(progress["finished"], ["first", "second", "evaluate"])
        last = self.store.objects[f"{PREFIX}/outputs/out/first/last.pt"]
        self.assertEqual(hashlib.sha256(last).hexdigest(), progress["files"]["out/first/last.pt"]["sha256"])
        self.assertEqual(self.provider.machines(), [])

    def test_rents_a_machine_runs_the_steps_keeps_their_files_and_returns_the_machine(self):
        outcome = self.run_to_end(self.controller())

        self.assertEqual(outcome, "complete")
        self.assert_results_kept()
        self.assertEqual(self.provider.rented, ["m1"])
        # Checkpoints are larger than one PUT takes, so they went up in parts.
        self.assertGreater(self.store.joined, 0)
        self.assert_no_url_written()

    def test_replaces_a_machine_that_stops_answering_and_resumes_from_its_last_kept_checkpoint(self):
        def kill_on_first_checkpoint(key):
            if "/checkpoints/first/" in key and self.provider.rented == ["m1"]:
                self.provider.kill("m1")

        self.store.on_object = kill_on_first_checkpoint

        outcome = self.run_to_end(self.controller())

        self.assertEqual(outcome, "complete")
        self.assert_results_kept()
        self.assertEqual(self.provider.rented, ["m1", "m2"])
        self.assertIn(("m1", "no answer: Unreachable"), self.provider.notes)
        start = json.loads(self.machine_file("m2", ".agent/logs/first.log").read_text().splitlines()[0])
        self.assertGreaterEqual(start["resumed_from"], 10)
        self.assert_no_url_written()

    def test_waits_while_no_offer_fits_and_rents_once_one_does(self):
        self.provider.offers = [{"price": 0.9, "speed": 1.0, "count": 5}]
        later = threading.Timer(1.0, lambda: self.provider.offers.append({"price": 0.5, "speed": 1.0, "count": 1}))
        later.start()

        outcome = self.run_to_end(self.controller())

        later.cancel()
        self.assertEqual(outcome, "complete")
        names = [event["event"] for event in self.events()]
        self.assertLess(names.index("waiting"), names.index("rented"))
        self.assertEqual(self.provider.rented, ["m1"])
        self.assert_results_kept()

    def test_replaces_a_machine_that_trains_much_slower_than_expected_after_it_saves_its_checkpoint(self):
        self.provider.offers = [{"price": 0.4, "speed": 0.25, "count": 1}, {"price": 0.5, "speed": 1.0, "count": 1}]

        outcome = self.run_to_end(self.controller())

        self.assertEqual(outcome, "complete")
        self.assertEqual(self.provider.rented, ["m1", "m2"])
        self.assertTrue(any(ident == "m1" and text.startswith("slow:") for ident, text in self.provider.notes))
        start = json.loads(self.machine_file("m2", ".agent/logs/first.log").read_text().splitlines()[0])
        self.assertGreater(start["resumed_from"], 0)
        self.assert_results_kept()

    def test_finishes_a_run_that_lasts_far_longer_than_any_url(self):
        controller = self.controller(url_seconds=1.0)
        began = time.monotonic()

        outcome = self.run_to_end(controller)

        self.assertEqual(outcome, "complete")
        self.assertGreater(time.monotonic() - began, 3)
        self.assert_results_kept()

    def test_carries_the_run_on_when_the_controller_stops_and_starts_again(self):
        first = self.controller()
        first.run(stop=lambda: "first" in first.progress["finished"])

        outcome = self.run_to_end(self.controller())

        self.assertEqual(outcome, "complete")
        self.assertEqual(self.provider.rented, ["m1"])
        self.assert_results_kept()

    def test_the_provider_returns_the_machine_at_its_deadline_when_no_controller_is_left(self):
        controller = self.controller(lease_hours=1.5 / 3600, renew_hours=0.5 / 3600)
        controller.run(stop=lambda: any(event["event"] == "runner_started" for event in self.events()))

        time.sleep(2.5)

        self.assertEqual(self.provider.machines(), [])
        self.assertEqual(self.provider.machines_["m1"]["returned"], "deadline")

    def test_keeps_the_log_and_checkpoint_of_a_failed_step_returns_the_machine_and_stops(self):
        self.spec["steps"][0]["command"] += ["--fail-at", "25"]

        outcome = self.run_to_end(self.controller())

        self.assertEqual(outcome, "failed")
        self.assertIn(f"{PREFIX}/logs/first.log", self.store.objects)
        self.assertIn(f"{PREFIX}/checkpoints/first/last.pt", self.store.objects)
        self.assertEqual(self.provider.machines(), [])
        self.assertEqual(self.provider.rented, ["m1"])
        self.assertEqual(self.events()[-1]["outcome"], "failed")
        self.assert_no_url_written()

    def test_the_fake_store_refuses_an_expired_url_and_a_put_over_its_limit(self):
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        expired = self.store.presign_get("data/pretrain/corpus.bin", -1)
        with self.assertRaises(HTTPError):
            urlopen(expired, timeout=10)
        large = Request(self.store.presign_put(f"{PREFIX}/large", 30), data=b"x" * (64 * 1024 + 1), method="PUT")
        with self.assertRaises(HTTPError):
            urlopen(large, timeout=10)
        self.assertEqual([reason for _, reason in self.store.refused], ["expired", "too large"])


if __name__ == "__main__":
    unittest.main()
