"""Run Condition B: LoRA fine-tuning for TaigiSpeech."""

import argparse
from pathlib import Path

import wandb
from torch.utils.data import DataLoader, Subset

from data import EMERGENCY_INTENTS, build_datasets, collate_fn
from models import BACKBONE_NAME, build_model_b, count_parameters
from train import train
from utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Condition B with patched WavLM LoRA attention."
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)

    parser.add_argument("--no_wandb", action="store_true")

    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/B_lora_patched_best.pt",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="B_lora_patched",
    )
    parser.add_argument(
        "--artifact_name",
        type=str,
        default="taigispeech-condition-b-patched",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="taigispeech-hierarchical",
    )

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Required by the effective-weight attention implementation.
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if args.lora_r < 1:
        raise ValueError("--lora_r must be at least 1.")
    if args.lora_dropout != 0.0:
        raise ValueError(
            "The patched attention implementation requires "
            "--lora_dropout 0.0."
        )

    seed = 42
    set_seed(seed)

    device = get_device()
    print(f"Device: {device}")

    splits, label_names, label2id = build_datasets()

    emergency_indices = sorted(
        label2id[name]
        for name in EMERGENCY_INTENTS
        if name in label2id
    )

    print(f"Labels: {label_names}")
    print(f"Emergency IDs: {emergency_indices}")
    print(
        "Official split sizes: "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

    # Never use the test split for checkpoint selection.
    train_ds = splits["train"]
    val_ds = splits["val"]

    if args.limit_train is not None:
        train_ds = Subset(
            train_ds,
            list(range(min(args.limit_train, len(train_ds)))),
        )
        print(f"Limited training set: {len(train_ds)} samples")

    if args.limit_val is not None:
        val_ds = Subset(
            val_ds,
            list(range(min(args.limit_val, len(val_ds)))),
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
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = build_model_b(
        num_labels=len(label_names),
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    parameter_stats = count_parameters(model)

    print("\nLoRA model parameter summary")
    print(f"Total parameters: {parameter_stats['total_parameters']:,}")
    print(
        "Trainable parameters: "
        f"{parameter_stats['trainable_parameters']:,}"
    )
    print(
        "Trainable percentage: "
        f"{parameter_stats['trainable_percentage']:.2f}%"
    )

    model.print_trainable_parameters()

    # Explicitly record the attention patch and dropout setting.
    wandb_run = None

    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "condition": "B",
                "backbone": BACKBONE_NAME,
                "method": "LoRA",
                "attention_implementation": (
                    "WavLM effective-weight LoRA patch"
                ),
                "transformers_version": "4.57.6",
                "target_modules": ["q_proj", "v_proj"],
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "modules_to_save": ["projector", "classifier"],
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": seed,
                "selection_metric": "validation_macro_f1",
                **parameter_stats,
            },
        )

    checkpoint_path = Path(args.ckpt)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        best_epoch, best_f1 = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            emergency_indices=emergency_indices,
            epochs=args.epochs,
            lr=args.lr,
            ckpt_path=str(checkpoint_path),
            wandb_run=wandb_run,
            artifact_name=args.artifact_name,
            condition_name="B",
            artifact_description=(
                "Best validation-macro-F1 checkpoint for TaigiSpeech "
                "Condition B: WavLM-base-plus with LoRA-adjusted Q/V "
                "attention weights and a trainable classification head. "
                "Uses a Transformers 4.57.6 attention compatibility patch "
                "and lora_dropout=0.0."
            ),
        )

        if wandb_run is not None:
            wandb_run.summary["best_val_macro_f1"] = best_f1
            wandb_run.summary["best_epoch_index_zero_based"] = best_epoch
            wandb_run.summary["best_epoch_number_one_based"] = (
                best_epoch + 1
            )
            wandb_run.summary["checkpoint_path"] = str(checkpoint_path)

        print(f"\nBest validation macro-F1: {best_f1:.4f}")
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Checkpoint: {checkpoint_path.resolve()}")

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

