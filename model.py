"""A small character GPT: scalar autodiff, causal attention, Adam, and sampling.

Learning reference: Andrej Karpathy's microgpt
https://karpathy.github.io/2026/02/12/microgpt/
"""

import argparse
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
NAMES_SHA256 = "0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d"


class Scalar:
    """A number with the local derivatives needed for backpropagation."""

    __slots__ = ("grad", "parents", "value")

    def __init__(self, value, parents=()):
        self.value = float(value)
        self.grad = 0.0
        self.parents = parents

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return Scalar(self.value + other.value, ((self, 1.0), (other, 1.0)))

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return Scalar(
            self.value * other.value, ((self, other.value), (other, self.value))
        )

    __rmul__ = __mul__

    def __pow__(self, exponent):
        return Scalar(
            self.value**exponent,
            ((self, exponent * self.value ** (exponent - 1)),),
        )

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        value = math.exp(self.value)
        return Scalar(value, ((self, value),))

    def log(self):
        return Scalar(math.log(self.value), ((self, 1.0 / self.value),))

    def relu(self):
        return Scalar(max(0.0, self.value), ((self, float(self.value > 0)),))

    def backward(self):
        ordered, visited = [], set()
        pending = [(self, False)]
        while pending:
            node, ready = pending.pop()
            if ready:
                ordered.append(node)
            elif node not in visited:
                visited.add(node)
                pending.append((node, True))
                pending.extend((parent, False) for parent, _ in node.parents)
        for node in ordered:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(ordered):
            for parent, derivative in node.parents:
                parent.grad += node.grad * derivative


@dataclass
class Tokenizer:
    characters: str

    def __post_init__(self):
        if not self.characters or len(set(self.characters)) != len(self.characters):
            raise ValueError("tokenizer characters must be nonempty and unique")
        self.ids = {character: index for index, character in enumerate(self.characters)}

    @property
    def boundary(self):
        # The same extra token marks both the start and the end of a document.
        return len(self.characters)

    @property
    def size(self):
        return len(self.characters) + 1

    def encode(self, text):
        unknown = set(text) - self.ids.keys()
        if unknown:
            raise ValueError(
                f"characters outside the training vocabulary: {sorted(unknown)!r}"
            )
        return [self.ids[character] for character in text]

    def decode(self, tokens):
        return "".join(
            self.characters[token] for token in tokens if token != self.boundary
        )


@dataclass(frozen=True)
class Config:
    width: int = 16
    heads: int = 4
    layers: int = 1
    context: int = 16

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError(
                "width, heads, layers, and context must be positive integers"
            )
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


def linear(vector, matrix):
    return [sum(a * b for a, b in zip(vector, row)) for row in matrix]


def normalize(vector):
    inverse_rms = (
        sum(value * value for value in vector) * (1 / len(vector)) + 1e-5
    ) ** -0.5
    return [value * inverse_rms for value in vector]


def softmax(logits):
    maximum = max(value.value for value in logits)
    exponentials = [(value - maximum).exp() for value in logits]
    inverse_sum = sum(exponentials) ** -1
    return [value * inverse_sum for value in exponentials]


class GPT:
    def __init__(self, tokenizer, config=None, seed=42):
        config = config if config is not None else Config()
        self.tokenizer, self.config = tokenizer, config
        rng = random.Random(seed)
        width = config.width
        shapes = {
            "token": (tokenizer.size, width),
            "position": (config.context, width),
            "output": (tokenizer.size, width),
        }
        for layer in range(config.layers):
            for name in ("query", "key", "value", "attention"):
                shapes[f"{layer}.{name}"] = (width, width)
            shapes[f"{layer}.expand"] = (4 * width, width)
            shapes[f"{layer}.contract"] = (width, 4 * width)
        self.weights = {
            name: [
                [Scalar(rng.gauss(0, 0.08)) for _ in range(columns)]
                for _ in range(rows)
            ]
            for name, (rows, columns) in shapes.items()
        }
        self.parameters = [
            value for matrix in self.weights.values() for row in matrix for value in row
        ]

    def new_cache(self):
        return [([], []) for _ in range(self.config.layers)]

    def forward(self, token, position, cache):
        """Predict the next token using only tokens already added to this cache."""
        weights, config = self.weights, self.config
        hidden = [
            a + b
            for a, b in zip(weights["token"][token], weights["position"][position])
        ]
        head_width = config.width // config.heads
        for layer, (keys, values) in enumerate(cache):
            normalized = normalize(hidden)
            query = linear(normalized, weights[f"{layer}.query"])
            keys.append(linear(normalized, weights[f"{layer}.key"]))
            values.append(linear(normalized, weights[f"{layer}.value"]))
            attended = []
            for start in range(0, config.width, head_width):
                end = start + head_width
                scores = [
                    sum(q * k for q, k in zip(query[start:end], key[start:end]))
                    * (head_width**-0.5)
                    for key in keys
                ]
                probabilities = softmax(scores)
                attended.extend(
                    sum(
                        probability * value[channel]
                        for probability, value in zip(probabilities, values)
                    )
                    for channel in range(start, end)
                )
            hidden = [
                a + b
                for a, b in zip(hidden, linear(attended, weights[f"{layer}.attention"]))
            ]
            expanded = linear(normalize(hidden), weights[f"{layer}.expand"])
            contracted = linear(
                [value.relu() ** 2 for value in expanded], weights[f"{layer}.contract"]
            )
            hidden = [a + b for a, b in zip(hidden, contracted)]
        return linear(normalize(hidden), weights["output"])

    def loss(self, document):
        tokens = [
            self.tokenizer.boundary,
            *self.tokenizer.encode(document),
            self.tokenizer.boundary,
        ]
        cache, losses = self.new_cache(), []
        for position in range(min(len(tokens) - 1, self.config.context)):
            logits = self.forward(tokens[position], position, cache)
            maximum = max(value.value for value in logits)
            shifted = [value - maximum for value in logits]
            # Log-sum-exp avoids taking log(0) when a target is very unlikely.
            losses.append(
                sum(value.exp() for value in shifted).log()
                - shifted[tokens[position + 1]]
            )
        return sum(losses) * (1 / len(losses))

    def generate(self, prefix="", max_new_tokens=None, temperature=0.8, seed=42):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if len(prefix) >= self.config.context:
            raise ValueError(
                f"prefix must be shorter than {self.config.context} characters"
            )
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max-new-tokens must be positive")
        limit = self.config.context if max_new_tokens is None else max_new_tokens
        tokens = [self.tokenizer.boundary, *self.tokenizer.encode(prefix)]
        cache, rng = self.new_cache(), random.Random(seed)
        generated = []
        for position in range(self.config.context):
            logits = self.forward(tokens[position], position, cache)
            if position < len(prefix):
                continue
            if temperature == 0:
                token = max(range(len(logits)), key=lambda index: logits[index].value)
            else:
                maximum = max(value.value for value in logits)
                probabilities = [
                    math.exp((value.value - maximum) / temperature) for value in logits
                ]
                token = rng.choices(range(len(logits)), weights=probabilities)[0]
            if token == self.tokenizer.boundary:
                break
            generated.append(token)
            tokens.append(token)
            if len(generated) >= limit:
                break
        return prefix + self.tokenizer.decode(generated)


def train_model(
    model, documents, steps=1000, learning_rate=0.01, seed=42, progress=None
):
    if not documents or steps <= 0:
        raise ValueError("training requires documents and positive steps")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    rng, documents = random.Random(seed), list(documents)
    first = [0.0] * len(model.parameters)
    second = [0.0] * len(model.parameters)
    losses = []
    for step in range(steps):
        if step % len(documents) == 0:
            rng.shuffle(documents)
        loss = model.loss(documents[step % len(documents)])
        if not math.isfinite(loss.value):
            raise ValueError("training loss is not finite; try a smaller learning rate")
        loss.backward()
        rate = learning_rate * (1 - step / steps)
        for index, parameter in enumerate(model.parameters):
            gradient = parameter.grad
            first[index] = 0.9 * first[index] + 0.1 * gradient
            second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
            mean = first[index] / (1 - 0.9 ** (step + 1))
            variance = second[index] / (1 - 0.999 ** (step + 1))
            parameter.value -= rate * mean / (math.sqrt(variance) + 1e-8)
            parameter.grad = 0.0
        losses.append(loss.value)
        if progress is not None:
            progress(step + 1, losses)
    return losses


def evaluate(model, documents):
    # Every predicted character (including an end token) gets equal weight.
    count, total = 0, 0.0
    for document in documents:
        length = min(len(document) + 1, model.config.context)
        total += model.loss(document).value * length
        count += length
    if not count:
        raise ValueError("evaluation requires at least one document")
    return total / count


def save_model(model, path):
    payload = {
        "version": 1,
        "config": asdict(model.config),
        "characters": model.tokenizer.characters,
        "weights": {
            name: [[value.value for value in row] for row in matrix]
            for name, matrix in model.weights.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def load_model(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["version"] != 1:
        raise ValueError("unsupported checkpoint version")
    model = GPT(Tokenizer(payload["characters"]), Config(**payload["config"]))
    if payload["weights"].keys() != model.weights.keys():
        raise ValueError("checkpoint weight names do not match its configuration")
    for name, matrix in model.weights.items():
        saved = payload["weights"][name]
        if len(saved) != len(matrix) or any(
            len(a) != len(b) for a, b in zip(saved, matrix)
        ):
            raise ValueError(f"checkpoint has the wrong dimensions for {name}")
        for row, saved_row in zip(matrix, saved):
            for parameter, value in zip(row, saved_row):
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"checkpoint contains an invalid weight in {name}")
                parameter.value = float(value)
    return model


def fetch_names(path):
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(NAMES_URL, timeout=30) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != NAMES_SHA256:
        raise ValueError(
            "names dataset checksum does not match; use a different output path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"dataset: {path} ({len(data.decode('utf-8').splitlines())} names)")


def train_command(args):
    documents = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(documents) < 2:
        raise ValueError(
            "input must contain at least two nonempty lines, one document per line"
        )
    if not 0 < args.validation_fraction < 1 or args.eval_docs <= 0:
        raise ValueError(
            "validation-fraction must be between 0 and 1; eval-docs must be positive"
        )
    random.Random(args.seed).shuffle(documents)
    validation_count = min(
        len(documents) - 1, max(1, int(len(documents) * args.validation_fraction))
    )
    validation = documents[:validation_count][: args.eval_docs]
    training = documents[validation_count:]
    model = GPT(
        Tokenizer("".join(sorted(set("".join(documents))))),
        Config(args.width, args.heads, args.layers, args.context),
        args.seed,
    )
    print(
        f"parameters: {len(model.parameters)} | vocabulary: {model.tokenizer.size}",
        flush=True,
    )
    print(
        f"training documents: {len(training)} | held-out documents: {validation_count}",
        flush=True,
    )
    print(
        f"validation sample: {len(validation)} documents | context: {args.context}",
        flush=True,
    )
    started = time.perf_counter()
    initial_loss = evaluate(model, validation)
    print(f"initial validation loss: {initial_loss:.4f}", flush=True)

    def progress(step, losses):
        if step == 1 or step % 50 == 0 or step == args.steps:
            recent = losses[-50:]
            print(
                f"step {step:4}/{args.steps} | mean training loss: {sum(recent) / len(recent):.4f}",
                flush=True,
            )

    losses = train_model(
        model, training, args.steps, args.learning_rate, args.seed, progress
    )
    final_loss = evaluate(model, validation)
    save_model(model, args.output / "model.json")
    metrics = {
        "seed": args.seed,
        "input": str(args.input),
        "config": asdict(model.config),
        "parameters": len(model.parameters),
        "training_documents": len(training),
        "held_out_documents": validation_count,
        "evaluated_documents": len(validation),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "training_losses": losses,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        f"final validation loss: {final_loss:.4f} | checkpoint: {args.output / 'model.json'}",
        flush=True,
    )
    print("samples:", flush=True)
    for index in range(5):
        print(f"  {model.generate(seed=args.seed + index)!r}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CPUで学習・生成する小さな文字単位のGPT"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="名前のサンプルデータを取得する")
    fetch.add_argument("--output", type=Path, default=Path("data/names.txt"))
    train = commands.add_parser("train", help="1行1文書のテキストから学習する")
    train.add_argument("--input", type=Path, default=Path("data/names.txt"))
    train.add_argument("--output", type=Path, default=Path("runs/names"))
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--width", type=int, default=16)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=1)
    train.add_argument("--context", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--validation-fraction", type=float, default=0.1)
    train.add_argument("--eval-docs", type=int, default=64)
    train.add_argument("--seed", type=int, default=42)
    generate = commands.add_parser(
        "generate", help="保存したモデルから文字列を生成する"
    )
    generate.add_argument(
        "--checkpoint", type=Path, default=Path("runs/names/model.json")
    )
    generate.add_argument("--prefix", default="")
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument("--max-new-tokens", type=int)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch_names(args.output)
        elif args.command == "train":
            train_command(args)
        else:
            if args.count <= 0:
                raise ValueError("count must be positive")
            model = load_model(args.checkpoint)
            for index in range(args.count):
                print(
                    model.generate(
                        args.prefix,
                        args.max_new_tokens,
                        args.temperature,
                        args.seed + index,
                    )
                )
    except (ValueError, KeyError, OSError, urllib.error.URLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
