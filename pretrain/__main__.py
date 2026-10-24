import argparse
import contextlib
import json
import logging
import os
import signal
import sys
from pathlib import Path

import torch

from .data import file_digest, open_corpus, prepare_data
from .training import (
    evaluate,
    evaluate_windows,
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
    recipe = commands.add_parser(
        "recipe",
        help="日英の学習計画を確認し、語彙・コーパス・指示学習の設定を準備する",
    )
    recipe.add_argument(
        "--config", type=Path, default=Path("configs/pretrain-500m.toml")
    )
    recipe.add_argument("--workdir", type=Path)
    recipe.add_argument(
        "--workers", type=int, default=1, help="コーパスを並列に作るプロセス数"
    )
    recipe.add_argument(
        "--stage",
        choices=("inspect", "tokenizer", "corpus", "select", "validate", "sft-config"),
        default="inspect",
    )
    recipe.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="候補の文書をモデルで採点する装置（selectの段階）",
    )
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
    benchmarks = commands.add_parser(
        "fetch-benchmarks",
        help="標準ベンチマークの問題を学習データの混入除去用に取得する",
    )
    benchmarks.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    instructions = commands.add_parser(
        "fetch-instructions", help="日英の指示データを出典付きで取得する"
    )
    instructions.add_argument(
        "--output", type=Path, default=Path("data/instructions-source")
    )
    tuning = commands.add_parser(
        "prepare-instructions",
        help="指示データをAPIと同じ会話形式にし、回答部分を学習対象として詰める",
    )
    tuning.add_argument("--source", type=Path, default=Path("data/instructions-source"))
    tuning.add_argument("--tokenizer", type=Path, required=True)
    tuning.add_argument("--output", type=Path, default=Path("data/instructions"))
    tuning.add_argument("--context", type=int, default=2048)
    tuning.add_argument("--validation-conversations", type=int, default=1000)
    tuning.add_argument(
        "--tokens-per-language", type=int, help="各言語の学習用会話のトークン数の上限"
    )
    tuning.add_argument(
        "--benchmarks", type=Path, help="このベンチマークの問題を含む会話を除く"
    )
    tuning.add_argument("--seed", type=int, default=42)
    tuning.add_argument(
        "--max-reply-tokens",
        type=int,
        help="どれかの回答がこのトークン数を超える会話を除く",
    )
    preferences = commands.add_parser(
        "prepare-preferences",
        help="好ましい回答と好ましくない回答の組を会話形式にし、最後の回答を学習対象として並べる",
    )
    preferences.add_argument("--source", type=Path, required=True, help="言語ごとの組の JSON 行を置いた場所")
    preferences.add_argument("--tokenizer", type=Path, required=True)
    preferences.add_argument("--output", type=Path, required=True)
    preferences.add_argument("--context", type=int, default=2048)
    preferences.add_argument("--validation-pairs", type=int, default=500)
    preferences.add_argument("--benchmarks", type=Path, help="このベンチマークの問題を含む組を除く")
    preferences.add_argument("--seed", type=int, default=42)
    optimization = commands.add_parser(
        "train-preferences",
        help="チェックポイントから、好ましい回答を選ぶように DPO で学習する",
    )
    optimization.add_argument("--config", type=Path, required=True)
    optimization.add_argument("--data", type=Path, required=True)
    optimization.add_argument("--initialize-from", type=Path, required=True)
    optimization.add_argument("--output", type=Path, required=True)
    optimization.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    optimization.add_argument("--precision", default="auto")
    reinforcement = commands.add_parser(
        "train-reinforcement",
        help="チェックポイントから、答えを確かめられる問題で強化学習する（GRPO の系統）",
    )
    reinforcement.add_argument("--config", type=Path, required=True)
    reinforcement.add_argument("--prompts", type=Path, required=True, help="会話と答えの確かめ方を並べた JSON 行")
    reinforcement.add_argument("--initialize-from", type=Path, required=True)
    reinforcement.add_argument("--output", type=Path, required=True)
    reinforcement.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    reinforcement.add_argument("--precision", default="auto")
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
        "--continue-from",
        type=Path,
        help="保存した重み・最適化の状態・ステップを引き継ぎ、別のデータで学習を続ける",
    )
    training.add_argument(
        "--max-steps",
        type=int,
        help="学習計画を維持し、指定ステップで一旦保存して終了する",
    )
    training.add_argument("--threads", type=int, default=4)
    training.add_argument(
        "--compile",
        action="store_true",
        help="torch.compileで小さな演算をまとめて速くする（チェックポイントは同じ形）",
    )
    training.add_argument(
        "--hub-repo",
        default=os.environ.get("HF_MODEL_REPO"),
        help="学習段階の終了時にチェックポイントを保存するHFリポジトリ",
    )
    training.add_argument(
        "--hub-run-id", help="HFに保存する学習段階の名前（既定は出力ディレクトリ名）"
    )
    throughput = commands.add_parser(
        "throughput", help="学習設定の更新を合成データで流し、GPUでの学習速度を測る"
    )
    throughput.add_argument("--config", type=Path, required=True)
    throughput.add_argument("--vocabulary-size", type=int, required=True)
    throughput.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="試すマイクロバッチの大きさ（1ステップのトークン数は設定のまま）",
    )
    throughput.add_argument(
        "--compile", action="store_true", help="torch.compileでまとめた計算で測る"
    )
    throughput.add_argument("--output", type=Path, help="結果を保存するJSONファイル")
    evaluation = commands.add_parser(
        "evaluate", help="保存したモデルの検証損失を言語ごとに測る"
    )
    evaluation.add_argument("--data", type=Path, default=Path("data/bilingual"))
    evaluation.add_argument("--batches", type=int, default=16)
    evaluation.add_argument(
        "--every-window",
        action="store_true",
        help="検証データを先頭から文脈長ごとに区切り、すべての窓で測る",
    )
    evaluation.add_argument(
        "--max-windows", type=int, help="--every-windowで各言語に使う先頭からの窓の数"
    )
    evaluation.add_argument(
        "--other-data",
        action="store_true",
        help="学習に使ったものとは別の、同じ語彙のデータで測る",
    )
    evaluation.add_argument("--batch-size", type=int, default=8)
    generation = commands.add_parser(
        "generate", help="保存したモデルから文章を生成する"
    )
    generation.add_argument("--prompt", default="")
    generation.add_argument("--max-new-tokens", type=int, default=128)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-k", type=int, default=50)
    generation.add_argument("--seed", type=int, default=42)
    benchmark = commands.add_parser(
        "benchmark", help="保存したモデルをLM Evaluation Harnessの標準タスクで評価する"
    )
    benchmark.add_argument(
        "--tasks", nargs="+", help="評価するタスク名（既定は標準評価の全課題）"
    )
    benchmark.add_argument(
        "--limit", type=int, help="動作確認で各タスクに使う問題数（正式な評価は全問）"
    )
    benchmark.add_argument("--batch-tokens", type=int, default=4096)
    benchmark.add_argument("--threads", type=int, default=4)
    benchmark.add_argument(
        "--output", type=Path, required=True, help="問題ごとの記録を含む結果のJSONファイル"
    )
    hub_benchmark = commands.add_parser(
        "benchmark-hub", help="Hugging Face Hubのモデルを、標準評価と同じ課題と指標で評価する"
    )
    hub_benchmark.add_argument("--model", required=True, help="Hubのモデル名（例: openai-community/gpt2）")
    hub_benchmark.add_argument("--revision", required=True, help="評価するモデルのコミット")
    hub_benchmark.add_argument(
        "--tasks", nargs="+", help="評価するタスク名（既定は標準評価の全課題）"
    )
    hub_benchmark.add_argument(
        "--limit", type=int, help="動作確認で各タスクに使う問題数（正式な評価は全問）"
    )
    hub_benchmark.add_argument("--batch-size", type=int, default=32)
    hub_benchmark.add_argument("--threads", type=int, default=4)
    hub_benchmark.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    hub_benchmark.add_argument(
        "--output", type=Path, required=True, help="問題ごとの記録を含む結果のJSONファイル"
    )
    growth = commands.add_parser(
        "grow", help="学習したモデルの層を増やし、同じ予測から学習を始められる重みを作る"
    )
    growth.add_argument("--checkpoint", type=Path, required=True, help="元のモデルの重み")
    growth.add_argument("--output", type=Path, required=True, help="層を増やしたモデルの重み")
    growth.add_argument("--factor", type=int, default=2, help="層の数を何倍にするか")
    comparison = commands.add_parser(
        "compare-benchmarks",
        help="二つの標準評価の結果を、同じ問題ごとに言語別に比べる",
    )
    comparison.add_argument("--base", type=Path, required=True, help="比べる元の結果")
    comparison.add_argument("--other", type=Path, required=True, help="比べる先の結果")
    for command in (evaluation, generation, benchmark):
        command.add_argument(
            "--checkpoint", type=Path, default=Path("runs/bilingual/last.pt")
        )
        command.add_argument(
            "--device", choices=("cpu", "cuda", "auto"), default="auto"
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "recipe":
            from .recipe import (
                inspect_recipe,
                prepare_corpus,
                prepare_tokenizer,
                validate_corpus,
                write_sft_config,
            )

            workdir = args.workdir or Path("data") / args.config.stem
            if args.stage == "inspect":
                result = inspect_recipe(args.config, workdir)
            elif args.stage == "tokenizer":
                result = prepare_tokenizer(
                    args.config, workdir / "tokenizer", workdir / "benchmarks"
                )
            elif args.stage == "corpus":
                result = prepare_corpus(
                    args.config,
                    workdir / "tokenizer",
                    workdir,
                    workdir / "benchmarks",
                    args.workers,
                )
            elif args.stage == "select":
                from .selection import select_corpora

                result = select_corpora(
                    args.config, workdir, select_device(args.device)
                )
            elif args.stage == "validate":
                result = validate_corpus(args.config, workdir)
            else:
                result = write_sft_config(
                    args.config,
                    workdir / "instructions",
                    workdir / "tokenizer",
                    workdir / "sft.toml",
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "fetch-wikipedia":
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
        elif args.command == "fetch-benchmarks":
            from .benchmarks import fetch_benchmarks

            result = fetch_benchmarks(args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "benchmark-hub":
            from lm_eval.utils import setup_logging

            from .evaluation import evaluate_hub_benchmarks

            if args.threads <= 0:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
            # Show each task's steps on stderr while a long evaluation runs.
            setup_logging(logging.INFO)
            device = select_device(args.device)
            # The harness also prints notes; they join its log on stderr.
            with contextlib.redirect_stdout(sys.stderr):
                result = evaluate_hub_benchmarks(
                    args.model,
                    args.revision,
                    device,
                    select_precision(device),
                    args.batch_size,
                    args.tasks,
                    args.limit,
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result["by_language"], indent=2))
        elif args.command == "grow":
            from .growth import grow_checkpoint

            result = grow_checkpoint(args.checkpoint, args.output, args.factor)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "compare-benchmarks":
            from .comparison import compare_benchmarks

            result = compare_benchmarks(
                json.loads(args.base.read_text(encoding="utf-8")),
                json.loads(args.other.read_text(encoding="utf-8")),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "fetch-instructions":
            from .instructions import fetch_instructions

            result = fetch_instructions(args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "prepare-instructions":
            from .instructions import prepare_instructions

            result = prepare_instructions(
                args.source,
                args.output,
                args.tokenizer,
                args.context,
                args.validation_conversations,
                args.tokens_per_language,
                args.benchmarks,
                args.seed,
                args.max_reply_tokens,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "prepare-preferences":
            from .preference import prepare_preferences

            result = prepare_preferences(
                args.source,
                args.output,
                args.tokenizer,
                args.context,
                args.validation_pairs,
                args.benchmarks,
                args.seed,
            )
            print(json.dumps(result["pairs"], ensure_ascii=False, indent=2))
        elif args.command == "train-preferences":
            from .preference import read_config as read_preference_config
            from .preference import train_preferences

            result = train_preferences(
                args.data,
                args.output,
                args.initialize_from,
                read_preference_config(args.config),
                select_device(args.device),
                args.precision,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "train-reinforcement":
            from .reinforcement import read_config as read_reinforcement_config
            from .reinforcement import train_reinforcement

            result = train_reinforcement(
                args.prompts,
                args.output,
                args.initialize_from,
                read_reinforcement_config(args.config),
                select_device(args.device),
                args.precision,
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
        elif args.command == "throughput":
            import tomllib

            from .throughput import measure_training

            with args.config.open("rb") as source:
                settings = tomllib.load(source)
            result = measure_training(
                settings, args.vocabulary_size, args.batches, args.compile
            )
            text = json.dumps(result, indent=2)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n", encoding="utf-8")
            print(text)
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
                    args.continue_from,
                    args.compile,
                )
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
            if args.hub_repo:
                from .hub import publish_training_run

                result = publish_training_run(
                    args.output, args.hub_repo, args.hub_run_id
                )
                print(
                    json.dumps({"event": "hub_saved", **result}, ensure_ascii=False),
                    flush=True,
                )
        else:
            if args.command == "benchmark":
                if args.threads <= 0:
                    raise ValueError("threads must be positive")
                torch.set_num_threads(args.threads)
            device = select_device(args.device)
            model, tokenizer, payload = load_model(args.checkpoint, device)
            if args.command == "benchmark":
                from .evaluation import evaluate_benchmarks
                from lm_eval.utils import setup_logging

                # Show each task's steps on stderr while a long evaluation runs.
                setup_logging(logging.INFO)
                precision = select_precision(device)
                # The harness also prints notes; they join its log on stderr.
                with contextlib.redirect_stdout(sys.stderr):
                    result = evaluate_benchmarks(
                        model,
                        tokenizer,
                        device,
                        precision,
                        args.batch_tokens,
                        args.tasks,
                        args.limit,
                    )
                result["checkpoint"] = {
                    "checkpoint": str(args.checkpoint),
                    "sha256": file_digest(args.checkpoint),
                    "step": payload["step"],
                    "tokens_seen": payload["tokens_seen"],
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "precision": precision,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                # The file keeps every question; the summary is what to read.
                print(json.dumps(result["by_language"], indent=2))
            elif args.command == "evaluate":
                corpus = open_corpus(args.data, ("validation",))
                corpus.validate_context(model.config.context)
                if args.other_data:
                    if corpus.tokenizer.to_str() != tokenizer.to_str():
                        raise ValueError("evaluation data needs the model's tokenizer")
                elif corpus.fingerprint != payload["dataset_fingerprint"]:
                    raise ValueError(
                        "evaluation requires the checkpoint's held-out dataset"
                    )
                precision = select_precision(device)
                if args.max_windows is not None and not args.every_window:
                    raise ValueError("--max-windows applies to --every-window")
                if args.every_window:
                    if not hasattr(corpus, "streams"):
                        raise ValueError("--every-window needs a pretraining corpus")
                    result = evaluate_windows(
                        model,
                        corpus,
                        device,
                        precision,
                        args.batch_size,
                        args.max_windows,
                    )
                else:
                    result = evaluate(
                        model, corpus, device, precision, args.batches, args.batch_size
                    )
                result["checkpoint"] = {
                    "checkpoint": str(args.checkpoint),
                    "sha256": file_digest(args.checkpoint),
                    "step": payload["step"],
                    "tokens_seen": payload["tokens_seen"],
                }
                result["data"] = {
                    "path": str(args.data),
                    "fingerprint": corpus.fingerprint,
                    "every_window": args.every_window,
                }
                print(json.dumps(result, indent=2))
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
    except (
        ImportError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
