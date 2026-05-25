"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import math
    import sys
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from basics.vit import ViT
    from basics.text_encoder import FrozenTextEncoder
    from vlm.clip import ProjectionHeads, init_logit_scale, clip_loss
    from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES
    from vlm.eval import zeroshot_classification_accuracy
    import wandb

    device = args.device
    # 1. Data loaders
    train_loader, val_loader, test_loader = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 4),
    )

    # 2. Models
    vit = ViT(
        img_size=cfg["vit"]["img_size"],
        patch_size=cfg["vit"]["patch_size"],
        d_model=cfg["vit"]["d_model"],
        num_heads=cfg["vit"]["num_heads"],
        num_blocks=cfg["vit"]["num_blocks"],
        dropout=cfg["vit"].get("dropout", 0.1),
        pos_encoding=cfg["vit"].get("pos_encoding", "learned"),
        max_seq_len=cfg["vit"].get("max_seq_len", 196),
        rope_base=cfg["vit"].get("rope_base", 10_000.0),
    ).to(device)
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"])
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # 3. Projection heads and logit scale
    proj_heads = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"].get("d_proj", 256),
    ).to(device)
    logit_scale = torch.nn.Parameter(init_logit_scale().to(device))

    params = list(vit.parameters()) + list(proj_heads.parameters()) + [logit_scale]
    optimizer = AdamW(
        params,
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"].get("betas", (0.9, 0.95))),
        weight_decay=cfg["optim"].get("weight_decay", 0.1),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["num_epochs"] * len(train_loader),
        eta_min=0.0,
    )

    # W&B
    if args.wandb:
        wandb.init(project="clip-eurosat", config=cfg)

    best_val_acc = 0.0
    best_state = None
    num_epochs = cfg["train"]["num_epochs"]
    log_every = cfg["train"].get("log_every", 50)
    eval_every_epoch = cfg["train"].get("eval_every_epoch", 1)
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
    }

    def save_metrics() -> None:
        with open(args.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    for epoch in range(1, num_epochs + 1):
        vit.train()
        proj_heads.train()
        total_loss = 0.0
        total_batches = 0
        for i, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            with torch.no_grad():
                text_embeds = text_encoder(captions).to(device)
            image_embeds = vit(images)
            image_proj, text_proj = proj_heads(image_embeds, text_embeds)
            loss = clip_loss(image_proj, text_proj, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Clamp logit_scale
            logit_scale.data.clamp_(max=math.log(100.0))
            total_loss += loss.item()
            total_batches += 1
            if (i + 1) % log_every == 0:
                print(f"Epoch {epoch} [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": epoch * len(train_loader) + i})

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch} done. Avg train loss: {avg_loss:.4f}")
        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(avg_loss)
        if args.wandb:
            wandb.log({"train/avg_loss": avg_loss, "epoch": epoch})

        # Validation
        if epoch % eval_every_epoch == 0:
            vit.eval()
            proj_heads.eval()
            with torch.no_grad():
                class_prompts = [f"a satellite image of {cls}" for cls in EUROSAT_CLASSES]
                class_indices = list(range(len(class_prompts)))
                val_acc = zeroshot_classification_accuracy(
                    vit, proj_heads, text_encoder, val_loader,
                    class_prompts, class_indices, device
                )
            print(f"Epoch {epoch} val zero-shot acc: {val_acc:.4f}")
            if args.wandb:
                wandb.log({"val/zeroshot_acc": val_acc, "epoch": epoch})
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "vit": vit.state_dict(),
                    "proj_heads": proj_heads.state_dict(),
                    "logit_scale": logit_scale.detach().cpu(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                }
            metrics["val_acc"].append(val_acc)
            save_metrics()


    # Save best checkpoint
    if best_state is not None:
        torch.save(best_state, args.output_dir / "best.pt")
        print(f"Best model saved with val acc {best_val_acc:.4f}")
    else:
        save_metrics()

    # --- Extrapolation evaluation on upsampled images (96x96, patch_size=8, 12x12 patches) ---
    print("\nExtrapolation evaluation (96x96 images, 12x12 patches)...")
    # Build upsampled val_loader
    up_img_size = 96
    up_num_patches = (up_img_size // cfg["vit"]["patch_size"]) ** 2
    up_vit = ViT(
        img_size=up_img_size,
        patch_size=cfg["vit"]["patch_size"],
        d_model=cfg["vit"]["d_model"],
        num_heads=cfg["vit"]["num_heads"],
        num_blocks=cfg["vit"]["num_blocks"],
        dropout=cfg["vit"].get("dropout", 0.1),
        pos_encoding=cfg["vit"].get("pos_encoding", "learned"),
        max_seq_len=cfg["vit"].get("max_seq_len", 196),
        rope_base=cfg["vit"].get("rope_base", 10_000.0),
    ).to(device)
    # Load weights (except positional embedding if learned)
    state_dict = vit.state_dict()
    up_state_dict = up_vit.state_dict()
    # Interpolate positional embeddings if learned
    if cfg["vit"].get("pos_encoding", "learned") == "learned":
        # Interpolate patch pos_embed (exclude CLS)
        pos_embed = state_dict["pos_embed"]  # (1, old_num_patches+1, d_model)
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]  # (1, old_num_patches, d_model)
        old_side = int(patch_pos.shape[1] ** 0.5)
        new_side = int(up_num_patches ** 0.5)
        patch_pos = patch_pos.reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2)  # (1, d_model, old, old)
        patch_pos_up = torch.nn.functional.interpolate(
            patch_pos, size=(new_side, new_side), mode="bicubic", align_corners=False
        )
        patch_pos_up = patch_pos_up.permute(0, 2, 3, 1).reshape(1, new_side * new_side, -1)
        up_state_dict["pos_embed"] = torch.cat([cls_pos, patch_pos_up], dim=1)
        # Copy other weights
        for k in up_state_dict:
            if k != "pos_embed" and k in state_dict and up_state_dict[k].shape == state_dict[k].shape:
                up_state_dict[k] = state_dict[k]
        up_vit.load_state_dict(up_state_dict)
    else:
        # RoPE: just copy all matching weights
        for k in up_state_dict:
            if k in state_dict and up_state_dict[k].shape == state_dict[k].shape:
                up_state_dict[k] = state_dict[k]
        up_vit.load_state_dict(up_state_dict)

    # Build upsampled val_loader
    _, up_val_loader, _ = build_eurosat_loaders(
        img_size=up_img_size,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 4),
    )
    up_vit.eval()
    proj_heads.eval()
    with torch.no_grad():
        class_prompts = [f"a satellite image of {cls}" for cls in EUROSAT_CLASSES]
        class_indices = list(range(len(class_prompts)))
        up_val_acc = zeroshot_classification_accuracy(
            up_vit, proj_heads, text_encoder, up_val_loader,
            class_prompts, class_indices, device
        )
    print(f"Extrapolation val zero-shot acc: {up_val_acc:.4f}")


if __name__ == "__main__":
    main()
