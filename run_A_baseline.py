
"""
Run Condition A: full fine-tuning baseline for TaigiSpeech.

Example:
    python run_A_baseline.py --lr 3e-5

Optional local-only training:
    python run_A_baseline.py --lr 3e-5 --no_wandb

The best checkpoint is saved locally and, when W&B is enabled,
uploaded as a versioned model artifact whenever validation macro-F1
improves.
"""

import argparse
from pathlib import Path

import wandb
from torch.utils.data import DataLoader, Subset

from data import EMERGENCY_INTENTS, build_datasets, collate_fn
from models import build_model_a
from train import train
from utils import get_device, set_seed


def parse_args():
    """Parse command-line options for the Condition A experiment."""
    parser = argparse.ArgumentParser(
        description="Train the TaigiSpeech Condition A baseline."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B tracking and artifact uploads.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/A_best.pt",
        help="Path for the locally saved best checkpoint.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="A_baseline_full_ft_checkpointed",
        help="Name of this experiment in W&B.",
    )
    parser.add_argument(
        "--artifact_name",
        type=str,
        default="taigispeech-condition-a",
        help="Stable W&B artifact name; improvements create new versions.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="taigispeech-hierarchical",
        help="W&B project name.",
    )
    return parser.parse_args()


def main():
    """Prepare data, initialize tracking, and run the baseline."""
    args = parse_args()

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

    train_ds = splits["train"]
    if args.limit_train is not None:
        train_ds = Subset(
            train_ds,
            list(range(min(args.limit_train, len(train_ds)))),
        )
        print(f"Smoke test: train limited to {len(train_ds)} samples")

    val_ds = splits["val"]
    if args.limit_val is not None:
        val_ds = Subset(
            val_ds,
            list(range(min(args.limit_val, len(val_ds)))),
        )
        print(f"Smoke test: validation limited to {len(val_ds)} samples")

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

    print(
        f"Train batches/epoch: {len(train_loader)} | "
        f"validation batches/epoch: {len(val_loader)}"
    )

    model = build_model_a(num_labels=len(label_names))

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "condition": "A",
                "backbone": "microsoft/wavlm-base-plus",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": seed,
                "selection_metric": "validation_macro_f1",
                "checkpoint_artifact": args.artifact_name,
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
        )

        if wandb_run is not None:
            wandb_run.summary["best_val_macro_f1"] = best_f1
            wandb_run.summary["best_epoch_index_zero_based"] = best_epoch
            wandb_run.summary["best_epoch_number_one_based"] = best_epoch + 1
            wandb_run.summary["checkpoint_path"] = str(checkpoint_path)

        print(f"Best validation macro-F1: {best_f1:.4f}")
        print(f"Best epoch (counting from 1): {best_epoch + 1}")
        print(f"Local checkpoint path: {checkpoint_path.resolve()}")

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()