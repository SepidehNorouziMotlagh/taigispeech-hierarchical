
"""
run_B_lora.py

Condition B: LoRA fine-tuning for TaigiSpeech intent classification.

Uses:
- WavLM-base-plus backbone
- LoRA adapters on q_proj and v_proj
- Trainable projector and classifier
- Official speaker-disjoint train/validation splits
- Validation macro-F1 for checkpoint selection
- Emergency recall and emergency F1 for evaluation
- Optional Weights & Biases logging

The official test split is never used for model selection.

IMPORTANT:
The patched models.py implementation computes effective LoRA weights
inside WavLM attention. This runner performs a gradient check before
starting training and aborts if LoRA gradients are missing or invalid.
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Subset

from data import (
    BACKBONE_NAME,
    EMERGENCY_INTENTS,
    build_datasets,
    collate_fn,
)
from models import build_model_b, count_parameters
from train import train
from utils import get_device, set_seed


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Condition B: WavLM with LoRA on TaigiSpeech."
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # The effective-weight attention patch requires dropout = 0.
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/B_best.pt",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="B_lora",
    )
    parser.add_argument(
        "--artifact_name",
        type=str,
        default="taigispeech-condition-b",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="taigispeech-hierarchical",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )

    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")

    if args.lr <= 0:
        raise ValueError("--lr must be positive.")

    if args.weight_decay < 0:
        raise ValueError("--weight_decay cannot be negative.")

    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")

    if args.lora_r < 1:
        raise ValueError("--lora_r must be at least 1.")

    if args.lora_alpha < 1:
        raise ValueError("--lora_alpha must be at least 1.")

    if args.lora_dropout != 0.0:
        raise ValueError(
            "The current effective-weight WavLM attention patch "
            "requires --lora_dropout 0.0."
        )

    if args.limit_train is not None and args.limit_train < 1:
        raise ValueError("--limit_train must be at least 1.")

    if args.limit_val is not None and args.limit_val < 1:
        raise ValueError("--limit_val must be at least 1.")


# ---------------------------------------------------------------------
# LoRA gradient verification
# ---------------------------------------------------------------------

def verify_lora_gradients(model, train_loader, device):
    """
    Run one diagnostic forward/backward pass without an optimizer step.

    This checks that:
    1. LoRA parameters exist.
    2. LoRA parameters participate in the computation graph.
    3. LoRA gradients are finite.
    4. At least one LoRA B matrix receives a nonzero gradient.

    A zero gradient for an A matrix at initialization can be normal:
    standard LoRA initializes B to zero. Missing gradients (None),
    however, indicate that the parameters are disconnected.
    """
    print("\n" + "=" * 65)
    print("PRE-TRAINING LoRA GRADIENT CHECK")
    print("=" * 65)

    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    try:
        batch = next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError(
            "Cannot verify gradients: training loader is empty."
        ) from exc

    input_values = batch["input_values"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["intent"].to(device)

    outputs = model(
        input_values=input_values,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"LoRA gradient check failed: non-finite loss {loss.item()}."
        )

    loss.backward()

    lora_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]

    if not lora_parameters:
        raise RuntimeError(
            "No LoRA parameters were found. "
            "Check build_model_b() and the PEFT configuration."
        )

    missing_gradients = []
    nonfinite_gradients = []
    nonzero_b_gradients = []
    zero_or_missing_norms = []

    for name, parameter in lora_parameters:
        gradient = parameter.grad

        if gradient is None:
            missing_gradients.append(name)
            continue

        if not torch.isfinite(gradient).all().item():
            nonfinite_gradients.append(name)
            continue

        grad_norm = gradient.detach().float().norm().item()

        if grad_norm == 0.0:
            zero_or_missing_norms.append(name)

        if "lora_B" in name and grad_norm > 0.0:
            nonzero_b_gradients.append((name, grad_norm))

    print(f"Diagnostic loss: {loss.item():.6f}")
    print(f"LoRA parameter tensors: {len(lora_parameters)}")
    print(
        "Parameters with gradients: "
        f"{len(lora_parameters) - len(missing_gradients)}"
        f"/{len(lora_parameters)}"
    )
    print(f"Missing gradients: {len(missing_gradients)}")
    print(f"Non-finite gradients: {len(nonfinite_gradients)}")
    print(f"Zero-norm gradient tensors: {len(zero_or_missing_norms)}")
    print(f"LoRA B tensors with nonzero gradients: {len(nonzero_b_gradients)}")

    if missing_gradients:
        print("\nExamples of parameters with missing gradients:")
        for name in missing_gradients[:10]:
            print("  ", name)

    if nonfinite_gradients:
        print("\nExamples of parameters with non-finite gradients:")
        for name in nonfinite_gradients[:10]:
            print("  ", name)

    if nonzero_b_gradients:
        print("\nExamples of nonzero LoRA B gradients:")
        for name, norm in nonzero_b_gradients[:5]:
            print(f"  {name}: norm={norm:.6e}")

    if missing_gradients:
        model.zero_grad(set_to_none=True)
        raise RuntimeError(
            "LoRA gradient check FAILED: some LoRA parameters have "
            "grad=None. Do not launch the full experiment."
        )

    if nonfinite_gradients:
        model.zero_grad(set_to_none=True)
        raise RuntimeError(
            "LoRA gradient check FAILED: non-finite gradients detected."
        )

    if not nonzero_b_gradients:
        model.zero_grad(set_to_none=True)
        raise RuntimeError(
            "LoRA gradient check FAILED: no LoRA B matrix received a "
            "nonzero gradient. Do not launch the full experiment."
        )

    # The diagnostic pass must not carry gradients into real training.
    model.zero_grad(set_to_none=True)

    print("\nPASS: LoRA gradients are connected and finite.")
    print("The diagnostic pass did not perform an optimizer step.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    validate_args(args)

    # Reproducibility.
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    print(f"Backbone: {BACKBONE_NAME}")
    print("Condition: B (LoRA)")
    print(f"Seed: {args.seed}")

    # -------------------------------------------------------------
    # Load the official dataset splits.
    # -------------------------------------------------------------

    splits, label_names, label2id = build_datasets()

    emergency_indices = sorted(
        label2id[name]
        for name in EMERGENCY_INTENTS
        if name in label2id
    )

    if len(label2id) != len(label_names):
        raise RuntimeError("Inconsistent label mapping.")

    missing_emergency = EMERGENCY_INTENTS - set(label2id)
    if missing_emergency:
        raise RuntimeError(
            "Emergency labels missing from dataset mapping: "
            f"{sorted(missing_emergency)}"
        )

    print("\nLabels:", label_names)
    print("Emergency IDs:", emergency_indices)

    print(
        "Official split sizes: "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

    # Use only train and validation for fitting/model selection.
    # Do not create or iterate a test loader here.
    train_ds = splits["train"]
    val_ds = splits["val"]

    if args.limit_train is not None:
        train_ds = Subset(
            train_ds,
            range(min(args.limit_train, len(train_ds))),
        )
        print(f"Limited training set: {len(train_ds)} samples")

    if args.limit_val is not None:
        val_ds = Subset(
            val_ds,
            range(min(args.limit_val, len(val_ds))),
        )
        print(f"Limited validation set: {len(val_ds)} samples")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Training and validation datasets must be nonempty."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # -------------------------------------------------------------
    # Build a fresh Condition B model.
    # -------------------------------------------------------------

    model = build_model_b(
        num_labels=len(label_names),
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    parameter_stats = count_parameters(model)

    # Current models.py is expected to return a dictionary.
    if not isinstance(parameter_stats, dict):
        raise TypeError(
            "count_parameters(model) must return a dictionary with "
            "'total_parameters', 'trainable_parameters', and "
            "'trainable_percentage' keys."
        )

    required_keys = {
        "total_parameters",
        "trainable_parameters",
        "trainable_percentage",
    }

    missing_keys = required_keys - set(parameter_stats)
    if missing_keys:
        raise KeyError(
            "count_parameters(model) is missing keys: "
            f"{sorted(missing_keys)}"
        )

    print("\nLoRA model parameter summary")
    print(f"Total parameters: {parameter_stats['total_parameters']:,}")
    print(
        "Trainable parameters: "
        f"{parameter_stats['trainable_parameters']:,}"
    )
    print(
        "Trainable percentage: "
        f"{parameter_stats['trainable_percentage']:.4f}%"
    )

    model.print_trainable_parameters()

    # -------------------------------------------------------------
    # Verify LoRA is connected before spending time on training.
    # -------------------------------------------------------------

    verify_lora_gradients(
        model=model,
        train_loader=train_loader,
        device=device,
    )

    # -------------------------------------------------------------
    # Initialize optional W&B logging.
    # -------------------------------------------------------------

    wandb_run = None

    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "condition": "B",
                "backbone": BACKBONE_NAME,
                "method": "LoRA",
                "target_modules": ["q_proj", "v_proj"],
                "modules_to_save": ["projector", "classifier"],
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "seed": args.seed,
                "train_size": len(train_ds),
                "validation_size": len(val_ds),
                "selection_metric": "validation_macro_f1",
                "emergency_indices": emergency_indices,
                **parameter_stats,
            },
        )

    checkpoint_path = Path(args.ckpt)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Train and select the best checkpoint using validation macro-F1.
    # -------------------------------------------------------------

    try:
        best_epoch, best_f1 = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            emergency_indices=emergency_indices,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            ckpt_path=str(checkpoint_path),
            wandb_run=wandb_run,
            artifact_name=args.artifact_name,
            condition_name="B",
            artifact_description=(
                "Best validation-macro-F1 checkpoint for TaigiSpeech "
                "Condition B: WavLM-base-plus with LoRA adapters on "
                "query/value attention projections and a trainable "
                "intent-classification head."
            ),
        )

        print("\n" + "=" * 65)
        print("CONDITION B TRAINING FINISHED")
        print("=" * 65)
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best validation macro-F1: {best_f1:.4f}")
        print(f"Checkpoint: {checkpoint_path.resolve()}")
        print("Official test split: NOT evaluated")

        if wandb_run is not None:
            wandb_run.summary["best_val_macro_f1"] = best_f1
            wandb_run.summary["best_epoch_index_zero_based"] = best_epoch
            wandb_run.summary["best_epoch_number_one_based"] = best_epoch + 1
            wandb_run.summary["checkpoint_path"] = str(checkpoint_path)

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
