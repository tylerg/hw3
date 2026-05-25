"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import cycle
from pathlib import Path

import torch
import yaml

# Add tqdm for progress bars
from tqdm import tqdm

import re

def normalize_answer(text: str) -> str:
    text = text.lower().strip()

    # remove common prefixes
    prefixes = [
        "answer:",
        "the answer is",
        "answer is",
    ]

    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    # keep only first token-ish answer
    text = re.split(r"[.,\n]", text)[0]
    text = text.strip()

    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
    # BUGFIX: Added missing --run-all flag referenced in main()
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Run all injection modes (cls, all_patches, interleaved)"
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_prompt(question: str, injection: str) -> str:
    prompt = f"Question: {question} Answer:"
    if injection == "interleaved":
        return f"<image> {prompt}"
    return prompt


def prepare_train_batch(
    tokenizer,
    questions: list[str],
    answers: list[str],
    injection: str,
    device: torch.device,
    max_length: int,
):
    prompts = [build_prompt(q, injection) for q in questions]
    full_texts = [f"{prompt} {answer}" for prompt, answer in zip(prompts, answers)]

    prompt_encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    full_encoding = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_encoding["input_ids"].to(device)
    attention_mask = full_encoding["attention_mask"].to(device)
    labels = input_ids.clone()
    prompt_lengths = prompt_encoding["attention_mask"].sum(dim=1)
    for batch_index, prompt_len in enumerate(prompt_lengths.tolist()):
        labels[batch_index, :prompt_len] = -100
    labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def build_vlm_inputs(
    model,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    injection: str,
    mask_mode: str,
):
    if injection == "cls":
        visual_features = model.vit(images)
    else:
        visual_features = model.vit(images, return_all_tokens=True)
    visual_embeds = model.projector(visual_features)

    dtype = next(model.decoder.parameters()).dtype

    visual_embeds = visual_embeds.to(dtype)

    text_embeds = model.decoder.get_input_embeddings()(input_ids)
    text_embeds = text_embeds.to(dtype)

    if injection == "interleaved":
        inputs_embeds, attention_mask, _, visual_ranges = model._build_interleaved_inputs(
            text_embeds=text_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            visual_embeds=visual_embeds,
        )
        if mask_mode == "image_bidir":
            attention_mask = model._build_interleaved_bidir_mask(
                attention_mask=attention_mask,
                visual_ranges=visual_ranges,
            )
    else:
        prefix_mask = torch.ones(
            visual_embeds.shape[:2],
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        if mask_mode == "image_bidir":
            attention_mask = model._build_prefix_bidir_mask(
                n_visual=visual_embeds.size(1),
                n_text=text_embeds.size(1),
                text_attention_mask=attention_mask[:, visual_embeds.size(1):],
                device=attention_mask.device,
                dtype=text_embeds.dtype,
            )
    return inputs_embeds, attention_mask


def evaluate_model(
    model,
    tokenizer,
    val_loader,
    injection: str,
    mask_mode: str,
    image_token_id: int | None,
    max_eval: int,
    generation_cfg: dict,
    device: torch.device,
) -> dict[str, float]:
    from vlm.eval import batch_clevr_accuracy

    model.vit.eval()
    model.projector.eval()
    model.decoder.eval()


    predictions: list[str] = []
    golds: list[str] = []
    q_types: list[str] = []
    processed = 0

    for batch in tqdm(val_loader, desc="Evaluating", total=max_eval):
        if processed >= max_eval:
            break

        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]
        q_types.extend(batch["q_type"])

        prompts = [build_prompt(q, injection) for q in questions]
        prompt_encoding = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=generation_cfg.get("max_prompt_length", 128),
        )
        input_ids = prompt_encoding["input_ids"].to(device)
        attention_mask = prompt_encoding["attention_mask"].to(device)

        inputs_embeds, gen_attention_mask = build_vlm_inputs(
            model=model,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            injection=injection,
            mask_mode=mask_mode,
        )

        position_ids = torch.arange(
            inputs_embeds.size(1),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(inputs_embeds.size(0), -1)

        print(gen_attention_mask.shape)
        
        generated = model.decoder.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=gen_attention_mask,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        predicted_ids = generated[:, input_ids.size(1):]

        batch_preds = tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )
        batch_preds = [normalize_answer(x) for x in batch_preds]
        if processed == 0:
            for i in range(min(5, len(batch_preds))):
                print("QUESTION:", questions[i])
                print("PRED:", repr(batch_preds[i]))
                print("GOLD:", repr(answers[i]))
                print()
        batch_preds = [normalize_answer(x) for x in batch_preds]
        predictions.extend(batch_preds)
        golds.extend([normalize_answer(x) for x in answers])
        processed += len(answers)

    return batch_clevr_accuracy(predictions[:max_eval], golds[:max_eval], q_types[:max_eval])


def save_metrics(metrics: dict, output_dir: Path) -> None:
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        base = Path("runs")
        if args.run_all:
            args.output_dir = base / f"vlm_all_{args.mask_mode}_{args.freeze_config}"
        else:
            args.output_dir = base / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    decoder_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.optim import AdamW
    from vlm.data import build_clevr_loaders
    from vlm.model import VisionLanguageModel
    from vlm.projector import VisionLanguageProjector
    from basics.vit import ViT

    def load_decoder_and_tokenizer(current_injection: str) -> tuple[object, object, int | None]:
        tokenizer = AutoTokenizer.from_pretrained(cfg["decoder"]["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if current_injection == "interleaved":
            tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        decoder = AutoModelForCausalLM.from_pretrained(
            cfg["decoder"]["model_name"],
            dtype=decoder_dtype,
            attn_implementation=cfg["decoder"]["attn_implementation"],
        ).to(device)
        if current_injection == "interleaved":
            decoder.resize_token_embeddings(len(tokenizer))
        image_token_id = None
        if current_injection == "interleaved":
            image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        return tokenizer, decoder, image_token_id

    def load_vit() -> torch.nn.Module:
        vit = ViT(
            img_size=cfg["vit"]["img_size"],
            patch_size=cfg["vit"]["patch_size"],
            d_model=cfg["vit"]["d_model"],
            num_heads=cfg["vit"]["num_heads"],
            num_blocks=cfg["vit"]["num_blocks"],
            dropout=cfg["vit"].get("dropout", 0.1),
        ).to(device)
        checkpoint = torch.load(args.pretrained_vit, map_location=device)
        vit.load_state_dict(checkpoint["vit"])
        vit.eval()
        for param in vit.parameters():
            param.requires_grad = False
        return vit

    train_loader, val_loader = build_clevr_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 4),
    )

    injection_modes = [args.injection]
    if args.run_all:
        injection_modes = ["cls", "all_patches", "interleaved"]


    for injection in injection_modes:
        experiment_dir = args.output_dir
        if args.run_all:
            experiment_dir = args.output_dir / injection
            experiment_dir.mkdir(parents=True, exist_ok=True)

        tokenizer, decoder, image_token_id = load_decoder_and_tokenizer(injection)
        vit = load_vit()
        projector = VisionLanguageProjector(
            d_image=cfg["vit"]["d_model"],
            d_decoder=decoder.config.hidden_size,
            expansion=cfg["projector"].get("expansion", 4),
        ).to(device)
        model = VisionLanguageModel(
            vit=vit,
            projector=projector,
            decoder=decoder,
            tokenizer=tokenizer,
            image_token_id=image_token_id,
        )

        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.vit.parameters():
            param.requires_grad = False
        for param in model.projector.parameters():
            param.requires_grad = True

        optimizer = AdamW(
            model.projector.parameters(),
            lr=cfg["optim"]["lr"],
            betas=tuple(cfg["optim"].get("betas", (0.9, 0.95))),
            weight_decay=cfg["optim"].get("weight_decay", 0.0),
        )

        gradient_accumulation_steps = cfg["train"].get("gradient_accumulation_steps", 1)
        num_steps = cfg["train"].get("num_steps", 2000)
        log_every = cfg["train"].get("log_every", 25)
        eval_every_steps = cfg["train"].get("eval_every_steps", 200)
        max_eval = cfg["train"].get("eval_max_examples", 500)
        max_length = cfg["train"].get("max_length", 128)

        model.projector.train()
        model.vit.eval()
        model.decoder.eval()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        best_val_acc = 0.0
        metrics = {
            "step": [],
            "train_loss": [],
            "val_accuracy": [],
            "peak_memory_bytes": [],
            "sec_per_step": [],
        }

        train_iter = cycle(train_loader)
        step = 0
        micro_step = 0
        total_loss = 0.0
        step_times: list[float] = []
        peak_memory = 0

        # Add tqdm progress bar for training steps
        for _ in tqdm(range(num_steps), desc=f"Training ({injection})"):
            batch = next(train_iter)
            images = batch["image"].to(device)
            input_ids, attention_mask, labels = prepare_train_batch(
                tokenizer,
                batch["question"],
                batch["answer"],
                injection,
                device,
                max_length,
            )

            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                injection=injection,
                mask_mode=args.mask_mode,
            )
            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            total_loss += loss.item() * gradient_accumulation_steps

            if micro_step % gradient_accumulation_steps == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_start = time.perf_counter()
                optimizer.step()
                optimizer.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_end = time.perf_counter()
                step += 1
                step_time = step_end - step_start
                step_times.append(step_time)
                if device.type == "cuda":
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

                if step % log_every == 0 or step == 1:
                    avg_loss = total_loss / step
                    print(
                        f"[{injection}] Step {step}/{num_steps} "
                        f"loss={avg_loss:.4f} sec_step={step_time:.4f} "
                        f"peak_mem={peak_memory / 1024**2:.1f} MiB"
                    )

                if step % eval_every_steps == 0 or step == num_steps:
                    val_metrics = evaluate_model(
                        model=model,
                        tokenizer=tokenizer,
                        val_loader=val_loader,
                        injection=injection,
                        mask_mode=args.mask_mode,
                        image_token_id=image_token_id,
                        max_eval=max_eval,
                        generation_cfg=cfg.get("generation", {}),
                        device=device,
                    )
                    val_acc = val_metrics["overall"]
                    print(f"[{injection}] Step {step} val_acc={val_acc:.4f}")
                    metrics["step"].append(step)
                    metrics["train_loss"].append(total_loss / step)
                    metrics["val_accuracy"].append(val_acc)
                    metrics["peak_memory_bytes"].append(peak_memory)
                    metrics["sec_per_step"].append(sum(step_times) / len(step_times))
                    save_metrics(metrics, experiment_dir)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(
                            {
                                "projector": model.projector.state_dict(),
                                "injection": injection,
                                "mask_mode": args.mask_mode,
                                "freeze_config": args.freeze_config,
                                "tokenizer": tokenizer.get_vocab(),
                                "val_acc": val_acc,
                                "step": step,
                            },
                            experiment_dir / "best.pt",
                        )

        print(f"[{injection}] Training complete. Best val acc: {best_val_acc:.4f}")
        tokens_per_example = 1 if injection == "cls" else model.vit.num_patches + 1
        print(f"[{injection}] Visual tokens per example: {tokens_per_example}")
        experiment_summary = {
            "best_val_accuracy": best_val_acc,
            "visual_tokens": tokens_per_example,
            "peak_memory_bytes": peak_memory,
            "avg_step_time_sec": sum(step_times) / len(step_times) if step_times else 0.0,
        }
        with open(experiment_dir / "summary.json", "w") as f:
            json.dump(experiment_summary, f, indent=2)

        print(f"Saved metrics to {experiment_dir / 'metrics.json'}")
        print(f"Saved best checkpoint to {experiment_dir / 'best.pt'}")
        print(f"Saved summary to {experiment_dir / 'summary.json'}")

    print("Training pipeline finished.")
    print("If run, outputs would be written under:")
    if args.run_all:
        print(f"  {args.output_dir}/<cls|all_patches|interleaved>")
    else:
        print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()