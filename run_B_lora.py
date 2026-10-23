"""Condition B: LoRA-adapted WavLM-base-plus (backbone frozen).

Same 8-class task, same training protocol, same lr and epoch budget as
Condition A. The only variable is the adaptation method: full fine-tuning
vs. LoRA. This tests whether parameter-efficient adaptation matches full
fine-tuning on a small speech dataset (~1,600 training samples).

Reference point: Condition A achieved 0.8332 val macro-F1 with 94.6M
trainable parameters. The question for B is whether ~0.5M trainable
parameters can reach a similar macro-F1.

Usage (from the project root; run on a Colab T4 for full training):

    # Full converged run, checkpoint + config persisted to Google Drive
    python run_B_lora.py --epochs 20 --save_config \\
        --ckpt /content/drive/MyDrive/taigispeech/checkpoints/B_best.pt

    # Quick smoke test (CPU ok; a few minutes on CPU)
    python run_B_lora.py --epochs 1 --limit_train 16 --limit_val 32 --no_wandb

LoRA hyperparameters are loaded from ``configs/lora_config.yaml`` and can be
overridden on the command line (e.g. ``--lora_r 16``).

The saved checkpoint is the LoRA adapter + classification head only
(~2-10 MB), not the 380 MB base backbone. It is meant to be published to
Hugging Face Hub in Phase 8.
"""

import argparse
import json
import os

import yaml
from torch.utils.data import DataLoader, Subset
import wandb

from data import build_datasets, collate_fn, EMERGENCY_INTENTS
from models import build_model_b
from train import train
from utils import get_device, set_seed


def default_run_name(args: argparse.Namespace, lora_r: int) -> str:
    """Auto-generate a W&B run name from condition + hyperparameters.

    Format: B_<mode>_r<rank>_lr<lr>_ep<epochs>_bs<batch_size>

    ``mode`` is "smoke" when either --limit_train or --limit_val is set.

    Examples:
        B_full_r8_lr3e-05_ep20_bs8   (converged LoRA baseline)
        B_full_r16_lr3e-05_ep20_bs8  (rank-override experiment)
        B_smoke_r8_lr3e-05_ep1_bs8   (16-sample smoke test)

    ``lora_r`` must be the RESOLVED rank (post-YAML, post-CLI-override),
    not ``args.lora_r`` — the CLI value may be None.
    """
    mode = (
        "smoke"
        if (args.limit_train is not None or args.limit_val is not None)
        else "full"
    )
    return (
        f"B_{mode}_r{lora_r}_lr{args.lr:.0e}"
        f"_ep{args.epochs}_bs{args.batch_size}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Condition B: LoRA-adapted WavLM-base-plus (backbone frozen).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs. Condition A peaked at epoch 17 of 20.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Per-step batch size. 8 fits a T4 with 15 s audio.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5,
        help="Learning rate. Matches Condition A for fair comparison. "
             "LoRA sometimes benefits from 1e-4; treat this as the "
             "controlled baseline.",
    )
    parser.add_argument(
        "--limit_train", type=int, default=None,
        help="If set, use only the first N train samples (smoke tests).",
    )
    parser.add_argument(
        "--limit_val", type=int, default=None,
        help="If set, use only the first N val samples (smoke tests).",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable W&B logging (use for smoke tests).",
    )
    parser.add_argument(
        "--ckpt", type=str, default="checkpoints/B_best.pt",
        help="Path for the best checkpoint. Point at Google Drive for "
             "persistence across Colab sessions.",
    )
    parser.add_argument(
        "--save_config", action="store_true",
        help="Write a JSON config alongside the checkpoint.",
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Override the auto-generated W&B run name.",
    )
    parser.add_argument(
        "--lora_config", type=str, default="configs/lora_config.yaml",
        help="Path to the LoRA YAML config.",
    )
    parser.add_argument(
        "--lora_r", type=int, default=None,
        help="Override LoRA rank from the YAML config.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve LoRA config BEFORE generating the run name, so the name
    # shows the actual rank rather than "None".
    # ------------------------------------------------------------------
    with open(args.lora_config) as f:
        lora_cfg = yaml.safe_load(f)

    lora_r = args.lora_r if args.lora_r is not None else lora_cfg["rank"]
    lora_alpha = lora_cfg["alpha"]
    lora_dropout = lora_cfg["dropout"]
    target_modules = lora_cfg["target_modules"]

    set_seed(42)
    device = get_device()
    run_name = args.run_name or default_run_name(args, lora_r)
    print(f"Device: {device}")
    print(f"Run name: {run_name}")
    print(
        f"LoRA: r={lora_r} alpha={lora_alpha} dropout={lora_dropout} "
        f"targets={target_modules}"
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
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
    print(
        f"train batches/epoch: {len(train_loader)} | "
        f"val batches/epoch: {len(val_loader)}"
    )

    # ------------------------------------------------------------------
    # Model — WavLM + LoRA (backbone frozen, adapters + head trainable)
    # ------------------------------------------------------------------
    model = build_model_b(
        num_labels=len(label_names),
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )

    # Trainable-parameter count is the LoRA diagnostic. With the fixes
    # above this should be roughly 0.4-0.6% of the backbone. If it shows
    # 100%, the backbone was not frozen (PEFT wrapper missing).
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[model] Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project="taigispeech-hierarchical",
            name=run_name,
            config={
                "condition": "B",
                "backbone": "microsoft/wavlm-base-plus",
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": target_modules,
                "trainable_params": trainable,
                "total_params": total,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "run_name": run_name,
            },
        )

    # ------------------------------------------------------------------
    # Checkpoint directory and config snapshot
    # ------------------------------------------------------------------
    ckpt_dir = os.path.dirname(args.ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    config_path = (
        args.ckpt.replace(".pt", "_config.json") if args.save_config else None
    )
    config_snapshot = {
        "condition": "B",
        "run_name": run_name,
        "backbone": "microsoft/wavlm-base-plus",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": target_modules,
        "trainable_params": trainable,
        "total_params": total,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "max_audio_seconds": 15,
        "label_names": label_names,
        "emergency_indices": emergency_indices,
    }

    # Written pre-training so hyperparameters are on disk even if the
    # process is killed mid-run.
    if config_path is not None:
        with open(config_path, "w") as f:
            json.dump(config_snapshot, f, indent=2)
        print(f"Config (pre-training) written to: {config_path}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_epoch, best_f1 = train(
        model, train_loader, val_loader, device,
        emergency_indices,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_path=args.ckpt,
        wandb_run=wandb_run,
    )

    # ------------------------------------------------------------------
    # Post-training: enrich config, finalize W&B
    # ------------------------------------------------------------------
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
