"""Plot CLIP training metrics and inspect zero-shot validation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads
from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot CLIP training metrics and inspect qualitative validation examples.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat/analysis"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-correct", type=int, default=5)
    p.add_argument("--num-incorrect", type=int, default=5)
    p.add_argument("--metrics-file", type=Path, default=None)
    return p.parse_args()


def unnormalize_image(img: torch.Tensor) -> torch.Tensor:
    img = img.cpu() * IMAGENET_STD.view(3, 1, 1) + IMAGENET_MEAN.view(3, 1, 1)
    return img.clamp(0.0, 1.0)


def plot_metrics(metrics_path: Path, output_dir: Path) -> None:
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    epochs = metrics["epoch"]
    train_loss = metrics.get("train_loss", [])
    val_acc = metrics.get("val_acc", [])

    output_dir.mkdir(parents=True, exist_ok=True)

    if train_loss:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_loss, marker="o")
        plt.title("CLIP train loss")
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "train_loss.png", dpi=200)
        plt.close()
        print(f"Saved train loss curve to {output_dir / 'train_loss.png'}")

    if val_acc:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, val_acc, marker="o")
        plt.title("CLIP zero-shot validation accuracy")
        plt.xlabel("epoch")
        plt.ylabel("val accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "val_accuracy.png", dpi=200)
        plt.close()
        print(f"Saved val accuracy curve to {output_dir / 'val_accuracy.png'}")


def load_clip_model(cfg: dict, checkpoint: Path, device: str):
    vit = ViT(
        img_size=cfg["vit"]["img_size"],
        patch_size=cfg["vit"]["patch_size"],
        d_model=cfg["vit"]["d_model"],
        num_heads=cfg["vit"]["num_heads"],
        num_blocks=cfg["vit"]["num_blocks"],
        dropout=cfg["vit"].get("dropout", 0.1),
    ).to(device)
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"])
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    proj_heads = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"].get("d_proj", 256),
    ).to(device)

    checkpoint_data = torch.load(checkpoint, map_location=device)
    vit.load_state_dict(checkpoint_data["vit"])
    proj_heads.load_state_dict(checkpoint_data["proj_heads"])
    return vit, text_encoder, proj_heads


def evaluate_val_set(
    vit: torch.nn.Module,
    text_encoder: FrozenTextEncoder,
    proj_heads: ProjectionHeads,
    val_loader,
    device: str,
) -> list[dict]:
    vit.eval()
    proj_heads.eval()
    class_prompts = [f"a satellite image of {cls}" for cls in EUROSAT_CLASSES]
    class_indices = list(range(len(class_prompts)))
    with torch.no_grad():
        class_text_embeds = text_encoder(class_prompts).to(device)
        _, class_proj = proj_heads(
            torch.zeros(len(class_prompts), vit.d_model, device=device),
            class_text_embeds,
        )
        class_proj = F.normalize(class_proj, dim=-1)

    all_results = []
    for images, captions in val_loader:
        images = images.to(device)
        labels = [class_prompts.index(c) for c in captions]
        feats = vit(images)
        img_proj, _ = proj_heads(feats, torch.zeros(images.size(0), text_encoder.embedding_dim, device=device))
        img_proj = F.normalize(img_proj, dim=-1)
        sims = img_proj @ class_proj.T
        topk = sims.topk(3, dim=-1)

        for i in range(images.size(0)):
            pred_ids = topk.indices[i].tolist()
            pred_scores = topk.values[i].tolist()
            top3 = [EUROSAT_CLASSES[idx] for idx in pred_ids]
            label_id = labels[i]
            all_results.append(
                {
                    "image": images[i].cpu(),
                    "caption": captions[i],
                    "label_id": label_id,
                    "label_name": EUROSAT_CLASSES[label_id],
                    "pred_ids": pred_ids,
                    "pred_scores": pred_scores,
                    "pred_names": top3,
                    "correct": pred_ids[0] == label_id,
                }
            )
    return all_results


def show_examples(results: list[dict], num_correct: int, num_incorrect: int, output_dir: Path) -> None:
    correct = [r for r in results if r["correct"]][:num_correct]
    incorrect = [r for r in results if not r["correct"]][:num_incorrect]
    selected = correct + incorrect
    if not selected:
        print("No examples found to plot.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(selected)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flat if n > 1 else [axes]
    for ax, example in zip(axes, selected):
        img = unnormalize_image(example["image"])
        ax.imshow(img.permute(1, 2, 0).numpy())
        status = "CORRECT" if example["correct"] else "WRONG"
        title = f"{status}\nGT: {example['label_name']}\nTop3: {', '.join(example['pred_names'])}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.tight_layout()
    fig_path = output_dir / "qualitative_examples.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved qualitative example grid to {fig_path}")


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_file is not None:
        plot_metrics(args.metrics_file, output_dir)
    else:
        metrics_path = args.checkpoint.parent / "metrics.json"
        if metrics_path.exists():
            plot_metrics(metrics_path, output_dir)
        else:
            print(f"No metrics file found at {metrics_path}. Skipping metric plots.")

    vit, text_encoder, proj_heads = load_clip_model(cfg, args.checkpoint, args.device)
    _, val_loader, _ = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"].get("batch_size", 256),
        num_workers=cfg["train"].get("num_workers", 4),
    )
    results = evaluate_val_set(vit, text_encoder, proj_heads, val_loader, args.device)
    show_examples(results, args.num_correct, args.num_incorrect, output_dir)

    # Save results for later analysis.
    results_path = output_dir / "val_analysis.jsonl"
    with open(results_path, "w") as f:
        for entry in results:
            json_entry = {
                "caption": entry["caption"],
                "label_id": entry["label_id"],
                "label_name": entry["label_name"],
                "pred_ids": entry["pred_ids"],
                "pred_scores": entry["pred_scores"],
                "pred_names": entry["pred_names"],
                "correct": entry["correct"],
            }
            f.write(json.dumps(json_entry) + "\n")
    print(f"Saved validation example metadata to {results_path}")


if __name__ == "__main__":
    main()
