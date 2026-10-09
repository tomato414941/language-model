import argparse
import json
import signal
from pathlib import Path

import torch

from .data import Corpus, prepare_data
from .training import (
    evaluate,
    load_model,
    read_config,
    select_device,
    select_precision,
    train,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="日本語・英語の言語モデルをゼロから学習する"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    wikipedia = commands.add_parser(
        "fetch-wikipedia", help="日本語・英語のWikipedia本文を出典付きで取得する"
    )
    wikipedia.add_argument("--output", type=Path, default=Path("data/wikipedia-source"))
    wikipedia.add_argument("--documents-per-language", type=int, default=20000)
    wikipedia.add_argument("--seed", type=int, default=42)
    web = commands.add_parser(
        "fetch-web", help="品質選別済みの日英Web本文をトークン量を指定して取得する"
    )
    web.add_argument("--output", type=Path, default=Path("data/web-source"))
    web.add_argument("--tokenizer", type=Path, required=True)
    web.add_argument("--tokens-per-language", type=int, default=500000000)
    web.add_argument("--wiki-source", type=Path)
    web.add_argument("--shards-per-language", type=int, default=64)
    web.add_argument("--workers", type=int, default=4)
    web.add_argument("--seed", type=int, default=42)
    prepare = commands.add_parser(
        "prepare", help="文書を重複除去し、BPEと学習・検証データを作る"
    )
    prepare.add_argument(
        "--japanese",
        type=Path,
        required=True,
        help="JSONLのtext欄、または空行で文書を区切ったUTF-8テキスト",
    )
    prepare.add_argument(
        "--english",
        type=Path,
        required=True,
        help="JSONLのtext欄、または空行で文書を区切ったUTF-8テキスト",
    )
    prepare.add_argument("--output", type=Path, default=Path("data/bilingual"))
    prepare.add_argument("--vocab-size", type=int, default=16384)
    prepare.add_argument("--validation-fraction", type=float, default=0.05)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument(
        "--tokenizer", type=Path, help="既存の自作語彙を使って追加データを準備する"
    )
    prepare.add_argument(
        "--tokenizer-bytes-per-language",
        type=int,
        default=8 * 1024 * 1024,
        help="語彙学習に使う各言語の本文量の上限（UTF-8バイト数）",
    )
    prepare.add_argument(
        "--provenance", type=Path, help="入力本文の出典とチェックサムを記録したJSON"
    )
    training = commands.add_parser(
        "train", help="学習し、再開可能なチェックポイントを保存する"
    )
    training.add_argument("--data", type=Path, default=Path("data/bilingual"))
    training.add_argument("--output", type=Path, default=Path("runs/bilingual"))
    training.add_argument("--config", type=Path, default=Path("configs/gpu.toml"))
    training.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    training.add_argument(
        "--precision",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    training.add_argument("--resume", type=Path, nargs="?", const=Path("last.pt"))
    training.add_argument(
        "--initialize-from",
        type=Path,
        help="保存したモデルの重みから新しい学習段階を開始する",
    )
    training.add_argument(
        "--max-steps",
        type=int,
        help="学習計画を維持し、指定ステップで一旦保存して終了する",
    )
    training.add_argument("--threads", type=int, default=4)
    evaluation = commands.add_parser(
        "evaluate", help="保存したモデルの検証損失を言語ごとに測る"
    )
    evaluation.add_argument("--data", type=Path, default=Path("data/bilingual"))
    evaluation.add_argument("--batches", type=int, default=16)
    evaluation.add_argument("--batch-size", type=int, default=8)
    generation = commands.add_parser(
        "generate", help="保存したモデルから文章を生成する"
    )
    generation.add_argument("--prompt", default="")
    generation.add_argument("--max-new-tokens", type=int, default=128)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-k", type=int, default=50)
    generation.add_argument("--seed", type=int, default=42)
    for command in (evaluation, generation):
        command.add_argument(
            "--checkpoint", type=Path, default=Path("runs/bilingual/best.pt")
        )
        command.add_argument(
            "--device", choices=("cpu", "cuda", "auto"), default="auto"
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-wikipedia":
            from .wikipedia import fetch_wikipedia

            print(
                json.dumps(
                    fetch_wikipedia(
                        args.output, args.documents_per_language, args.seed
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "fetch-web":
            from .web import fetch_web

            result = fetch_web(
                args.output,
                args.tokenizer,
                args.tokens_per_language,
                args.wiki_source,
                args.shards_per_language,
                args.workers,
                args.seed,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "prepare":
            result = prepare_data(
                args.japanese,
                args.english,
                args.output,
                args.vocab_size,
                args.validation_fraction,
                args.seed,
                args.provenance,
                args.tokenizer_bytes_per_language,
                args.tokenizer,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "train":
            if args.threads <= 0:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
            model_config, config = read_config(args.config)
            resume = (
                args.output / "last.pt"
                if args.resume == Path("last.pt")
                else args.resume
            )
            stopped = False

            def stop(signum, frame):
                nonlocal stopped
                stopped = True

            previous = {
                s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                train(
                    args.data,
                    args.output,
                    model_config,
                    config,
                    select_device(args.device),
                    args.precision,
                    resume,
                    args.max_steps,
                    lambda: stopped,
                    args.initialize_from,
                )
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
        else:
            device = select_device(args.device)
            model, tokenizer, payload = load_model(args.checkpoint, device)
            if args.command == "evaluate":
                corpus = Corpus(args.data)
                corpus.validate_context(model.config.context)
                if corpus.fingerprint != payload["dataset_fingerprint"]:
                    raise ValueError(
                        "evaluation requires the checkpoint's held-out dataset"
                    )
                print(
                    json.dumps(
                        evaluate(
                            model,
                            corpus,
                            device,
                            select_precision(device),
                            args.batches,
                            args.batch_size,
                        ),
                        indent=2,
                    )
                )
            else:
                print(
                    model.generate(
                        tokenizer,
                        args.prompt,
                        args.max_new_tokens,
                        args.temperature,
                        args.top_k,
                        args.seed,
                    )
                )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
