"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import math
    import time
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from basics.vit import ViT
    from basics.lora import apply_lora_to_attention
    from vlm.data import build_resisc45_loaders

    device = torch.device(args.device)
    num_classes = cfg["num_classes"]
    vit_cfg = cfg.get("vit", {})
    img_size = vit_cfg.get("img_size", 64)
    patch_size = vit_cfg.get("patch_size", 8)
    d_model = vit_cfg.get("d_model", 384)
    num_heads = vit_cfg.get("num_heads", 6)
    num_blocks = vit_cfg.get("num_blocks", 6)
    dropout = vit_cfg.get("dropout", 0.0)

    train_loader, test_loader = build_resisc45_loaders(
        img_size=img_size,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 4),
    )

    vit = ViT(
        img_size=img_size,
        patch_size=patch_size,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)
    classifier = torch.nn.Linear(d_model, num_classes).to(device)

    if args.method == "linear_probe":
        for param in vit.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    elif args.method == "lora":
        vit = apply_lora_to_attention(vit, args.rank, args.alpha)
        for name, param in vit.named_parameters():
            if name.endswith(".A") or name.endswith(".B"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable_params = (
            sum(p.numel() for p in vit.parameters() if p.requires_grad)
            + sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        )
    elif args.method == "full_ft":
        for param in vit.parameters():
            param.requires_grad = True
        trainable_params = (
            sum(p.numel() for p in vit.parameters() if p.requires_grad)
            + sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    method_cfg = cfg.get("methods", {}).get(args.method, {})
    lr = float(method_cfg.get("lr", cfg["optim"]["lr"]))
    weight_decay = cfg["optim"].get("weight_decay", 0.0)
    betas = tuple(cfg["optim"].get("betas", (0.9, 0.999)))

    optimizer = AdamW(
        [p for p in list(vit.parameters()) + list(classifier.parameters()) if p.requires_grad],
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["num_epochs"] * len(train_loader),
        eta_min=0.0,
    )

    def evaluate(loader: torch.utils.data.DataLoader) -> float:
        vit.eval()
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                feats = vit(images)
                logits = classifier(feats)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / max(total, 1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metrics = {
        "method": args.method,
        "rank": args.rank,
        "alpha": args.alpha,
        "trainable_params": trainable_params,
        "epoch": [],
        "train_loss": [],
        "test_acc": [],
        "peak_memory_bytes": 0,
        "wall_time_seconds": 0.0,
    }

    start_time = time.perf_counter()
    best_acc = 0.0
    best_state = None
    num_epochs = cfg["train"]["num_epochs"]
    log_every = cfg["train"].get("log_every", 25)
    eval_every_epoch = cfg["train"].get("eval_every_epoch", 1)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        vit.train()
        classifier.train()
        total_loss = 0.0
        total_batches = 0
        for i, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            feats = vit(images)
            logits = classifier(feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            total_batches += 1
            if i % log_every == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(total_batches, 1)
        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(avg_loss)
        print(f"Epoch {epoch} done. Avg train loss: {avg_loss:.4f}")

        if epoch % eval_every_epoch == 0:
            test_acc = evaluate(test_loader)
            metrics["test_acc"].append(test_acc)
            print(f"Epoch {epoch} test accuracy: {test_acc:.4f}")
            if test_acc > best_acc:
                best_acc = test_acc
                best_state = {
                    "vit": vit.state_dict(),
                    "classifier": classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                }

    end_time = time.perf_counter()
    metrics["wall_time_seconds"] = end_time - start_time
    if device.type == "cuda":
        metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    else:
        metrics["peak_memory_bytes"] = 0

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if best_state is not None:
        torch.save(best_state, args.output_dir / "best.pt")
        print(f"Saved best model with test acc {best_acc:.4f} to {args.output_dir / 'best.pt'}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
