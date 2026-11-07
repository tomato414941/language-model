"""A stand-in for training in the rental rehearsal.

It trains in steps at the speed its machine sets, writes the metrics a runner
reads, saves a checkpoint now and then and when told to stop, resumes from one,
and can fail at a given step.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

TOKENS_PER_STEP = 1000
SECONDS_PER_STEP = 0.02


def save(path, step, size):
    header = (json.dumps({"step": step}) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(header + os.urandom(max(0, size - len(header))))
    os.replace(temporary, path)


def saved_step(path):
    return json.loads(Path(path).read_bytes().split(b"\n", 1)[0])["step"]


def train(args):
    for given in (args.data, args.initialize_from):
        if given and not Path(given).exists():
            raise SystemExit(f"{given} is missing")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = saved_step(args.resume) if args.resume else 0
    speed = float(os.environ.get("REHEARSAL_SPEED", "1"))
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    print(json.dumps({"event": "start", "resumed_from": start}), flush=True)
    with (output / "metrics.jsonl").open("a") as metrics:
        for step in range(start + 1, args.steps + 1):
            time.sleep(SECONDS_PER_STEP / speed)
            metrics.write(json.dumps({"step": step, "tokens_seen": step * TOKENS_PER_STEP}) + "\n")
            metrics.flush()
            if step == args.fail_at:
                sys.exit(3)
            if stopping or step % args.checkpoint_every == 0 or step == args.steps:
                save(output / "last.pt", step, args.checkpoint_bytes)
            if stopping:
                return
    print(json.dumps({"event": "trained", "steps": args.steps}), flush=True)


def evaluate(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"checkpoint_step": saved_step(args.checkpoint)}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    training = commands.add_parser("train")
    training.add_argument("--output", required=True)
    training.add_argument("--steps", type=int, required=True)
    training.add_argument("--checkpoint-every", type=int, default=10)
    training.add_argument("--checkpoint-bytes", type=int, default=100 * 1024)
    training.add_argument("--data")
    training.add_argument("--initialize-from")
    training.add_argument("--resume")
    training.add_argument("--fail-at", type=int)
    training.set_defaults(function=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
