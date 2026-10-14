import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from pretrain.data import file_digest, open_corpus
from pretrain.training import (
    atomic_save,
    evaluate,
    load_checkpoint,
    load_model,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export and inspect a trained bilingual model"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--benchmark-tasks", nargs="+")
    args = parser.parse_args(argv)
    root, results = args.output, args.results
    results.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    selected = root / "best.pt" if (root / "best.pt").exists() else root / "last.pt"
    model, tokenizer, saved = load_model(selected, device)
    last = load_checkpoint(root / "last.pt")
    corpus = open_corpus(args.data)
    if corpus.fingerprint != saved["dataset_fingerprint"]:
        raise ValueError("inspection requires the checkpoint's held-out dataset")
    validation = evaluate(model, corpus, device, "bfloat16", batches=16, batch_size=4)
    benchmarks = None
    if args.benchmark_tasks is not None:
        from pretrain.evaluation import evaluate_benchmarks

        benchmarks = evaluate_benchmarks(
            model,
            tokenizer,
            device,
            "bfloat16",
            batch_tokens=16384,
            tasks=args.benchmark_tasks,
        )
    excluded = {"optimizer", "scaler", "sampling_rng", "torch_rng", "cuda_rng"}
    atomic_save(
        {key: value for key, value in saved.items() if key not in excluded},
        root / "model.pt",
    )
    summary = {
        "completed_step": last["step"],
        "tokens_seen": last["tokens_seen"],
        "selected_checkpoint_step": saved["step"],
        "selected_checkpoint_tokens_seen": saved["tokens_seen"],
        "validation": validation,
        "benchmarks": benchmarks,
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision": last["precision"],
        "skipped_updates": last["skipped_updates"],
        "model_bytes": (root / "model.pt").stat().st_size,
        "model_sha256": file_digest(root / "model.pt"),
        "dataset_fingerprint": saved["dataset_fingerprint"],
        "tokenizer_sha256": file_digest(args.data / "tokenizer.json"),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    if (
        selected != root / "last.pt"
        and saved["step"] == last["step"]
        and saved["validation"] == last["validation"]
        and all(
            torch.equal(value, last["model"][name])
            for name, value in saved["model"].items()
        )
    ):
        summary["best_checkpoint_uses_last"] = True
        selected.unlink()
    torch.set_num_threads(1)
    cpu_model, cpu_tokenizer, cpu_saved = load_model(
        root / "model.pt", torch.device("cpu")
    )
    if not all(
        torch.equal(value, saved["model"][name])
        for name, value in cpu_saved["model"].items()
    ):
        raise ValueError("exported weights differ from the selected checkpoint")
    prompt = "# Japan\nJapan is"
    text = cpu_model.generate(
        cpu_tokenizer, prompt, max_new_tokens=8, temperature=0, seed=42
    )
    if text != cpu_model.generate(
        cpu_tokenizer, prompt, max_new_tokens=8, temperature=0, seed=42
    ):
        raise ValueError("CPU generation is not reproducible")
    summary.update(
        cpu_inference_verified=True,
        cpu_verification_location="RunPod CPU",
        cpu_verification_threads=1,
        cpu_sample=text,
    )
    (results / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
