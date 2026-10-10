import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from pretrain.data import Corpus, file_digest
from pretrain.training import (
    atomic_save,
    autocast,
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
    args = parser.parse_args(argv)
    root, results = args.output, args.results
    results.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    selected = root / "best.pt" if (root / "best.pt").exists() else root / "last.pt"
    model, tokenizer, saved = load_model(selected, device)
    last = load_checkpoint(root / "last.pt")
    corpus = Corpus(args.data)
    if corpus.fingerprint != saved["dataset_fingerprint"]:
        raise ValueError("inspection requires the checkpoint's held-out dataset")
    validation = evaluate(model, corpus, device, "bfloat16", batches=16, batch_size=4)
    prompts = {
        "ja": ["# 日本\n日本は", "# 地球\n地球は", "# 機械学習\n機械学習は"],
        "en": [
            "# Japan\nJapan is",
            "# Earth\nThe Earth is",
            "# Machine learning\nMachine learning is",
        ],
    }
    samples = []
    for language, items in prompts.items():
        for prompt in items:
            with autocast(device, "bfloat16"):
                text = model.generate(tokenizer, prompt, max_new_tokens=96, seed=42)
            continuation = " ".join(text[len(prompt) :].split())
            grams = [
                continuation[i : i + 8] for i in range(max(0, len(continuation) - 7))
            ]
            samples.append(
                {
                    "language": language,
                    "prompt": prompt,
                    "text": text,
                    "repeated_character_8gram_fraction": 1
                    - len(set(grams)) / len(grams)
                    if grams
                    else 0,
                    "replacement_characters": continuation.count("\ufffd"),
                }
            )
    (results / "samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n"
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
