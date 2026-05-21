"""Sweep LoRA rank on RESISC45 and plot test accuracy vs rank."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep LoRA rank on RESISC45 and plot test accuracy.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/resisc_lora_rank_sweep"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--ranks", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    return p.parse_args()


def run_rank(rank: int, args: argparse.Namespace) -> float:
    rank_dir = args.output_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    alpha = 2 * rank

    cmd = [
        sys.executable,
        "scripts/finetune_resisc.py",
        "--config",
        str(args.config),
        "--method",
        "lora",
        "--rank",
        str(rank),
        "--alpha",
        str(alpha),
        "--pretrained",
        str(args.pretrained),
        "--output-dir",
        str(rank_dir),
        "--device",
        args.device,
    ]
    if args.batch_size is not None:
        # finetune_resisc.py does not currently support batch-size override via CLI,
        # so this is only here if the script is extended later.
        pass

    print(f"Running rank={rank}, alpha={alpha}")
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, check=True)

    metrics_path = rank_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    test_acc = None
    if metrics.get("test_acc"):
        test_acc = metrics["test_acc"][-1]
    elif metrics.get("best_test_acc") is not None:
        test_acc = metrics["best_test_acc"]
    else:
        raise ValueError(f"No test accuracy found in {metrics_path}")

    return test_acc


def plot_results(ranks: list[int], accuracies: list[float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, accuracies, marker="o")
    plt.title("LoRA rank sweep on RESISC45")
    plt.xlabel("LoRA rank r")
    plt.ylabel("Test accuracy")
    plt.xticks(ranks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / "lora_rank_sweep.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved rank sweep plot to {plot_path}")

    csv_path = output_dir / "lora_rank_sweep.csv"
    with open(csv_path, "w") as f:
        f.write("rank,test_acc\n")
        for rank, acc in zip(ranks, accuracies):
            f.write(f"{rank},{acc}\n")
    print(f"Saved rank sweep CSV to {csv_path}")


def main() -> None:
    args = parse_args()
    ranks = args.ranks
    accuracies = []
    for rank in ranks:
        acc = run_rank(rank, args)
        accuracies.append(acc)
        print(f"rank={rank} test_acc={acc:.4f}")

    plot_results(ranks, accuracies, args.output_dir)


if __name__ == "__main__":
    main()
