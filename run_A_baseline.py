import argparse
import json
import os

from torch.utils.data import DataLoader, Subset
import wandb

from data import build_datasets, collate_fn, EMERGENCY_INTENTS
from models import build_model_a
from train import train
from utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default="checkpoints/A_best.pt")
    parser.add_argument("--save_config", action="store_true",
                        help="Write a JSON config alongside the checkpoint")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    splits, label_names, label2id = build_datasets()
    emergency_indices = sorted(
        label2id[n] for n in EMERGENCY_INTENTS if n in label2id
    )
    print(f"Labels: {label_names}")
    print(f"Emergency ids: {emergency_indices}")

    train_ds = splits["train"]
    if args.limit_train is not None:
        train_ds = Subset(
            train_ds, list(range(min(args.limit_train, len(train_ds))))
        )
        print(f"Smoke test: train limited to {len(train_ds)} samples")

    val_ds = splits["val"]
    if args.limit_val is not None:
        val_ds = Subset(
            val_ds, list(range(min(args.limit_val, len(val_ds))))
        )
        print(f"Smoke test: val limited to {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"train batches/epoch: {len(train_loader)} | "
          f"val batches/epoch: {len(val_loader)}")

    model = build_model_a(num_labels=len(label_names))

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project="taigispeech-hierarchical",
            name="A_baseline_full_ft",
            config={
                "condition": "A",
                "backbone": "microsoft/wavlm-base-plus",
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
            },
        )

    ckpt_dir = os.path.dirname(args.ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    config_path = args.ckpt.replace(".pt", "_config.json") if args.save_config else None
    config_snapshot = {
        "condition": "A",
        "backbone": "microsoft/wavlm-base-plus",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "max_audio_seconds": 15,
        "label_names": label_names,
        "emergency_indices": emergency_indices,
    }
    if config_path is not None:
        with open(config_path, "w") as f:
            json.dump(config_snapshot, f, indent=2)
        print(f"Config (pre-training) written to: {config_path}")

    best_epoch, best_f1 = train(
        model, train_loader, val_loader, device,
        emergency_indices,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_path=args.ckpt,
        wandb_run=wandb_run,
    )

    if config_path is not None:
        config_snapshot["best_epoch"] = best_epoch
        config_snapshot["best_val_macro_f1"] = float(best_f1)
        with open(config_path, "w") as f:
            json.dump(config_snapshot, f, indent=2)
        print(f"Config (post-training) updated: {config_path}")

    if wandb_run is not None:
        wandb_run.summary["best_val_macro_f1"] = best_f1
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()


if __name__ == "__main__":
    main()
