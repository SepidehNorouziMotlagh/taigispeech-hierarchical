"""Condition A baseline: full fine-tuning of WavLM-base-plus on TaigiSpeech.

Trains a single 8-class intent classifier with no binary emergency head,
no LoRA adapters, and no consistency constraint. This is the reference
point that Conditions B through E are measured against — matching the
"Scratch (C=8)" column of Chang et al. (2025), Table 8.

Usage (from the project root; run on a Colab T4 for full training):

    # Full converged run, checkpoint + config persisted to Google Drive
    python run_A_baseline.py --epochs 20 --save_config \\
        --ckpt /content/drive/MyDrive/taigispeech/checkpoints/A_best.pt

    # Quick smoke test (CPU ok; ~30 s on T4, a few minutes on CPU)
    python run_A_baseline.py --epochs 1 --limit_train 16 --limit_val 32 --no_wandb

Hyperparameter defaults reflect empirical tuning on TaigiSpeech:

    lr=3e-5   1e-4 caused training collapse (constant prediction, loss=ln 8);
              1e-5 undertrained (still improving after 20 epochs)
    bs=8      Fits a T4 with 15 s audio; larger batches untested
    epochs=10 At 3e-5 the model is still improving at epoch 10 — use 20
              for a converged run (see research note for details)

Artifacts produced:

    --ckpt path             Best checkpoint (torch.save on every improvement;
                            written from inside train.py, not here)
    --ckpt with _config.json  Optional JSON snapshot of hyperparameters and
                            final metrics, written here when --save_config
    W&B run                 Project: taigispeech-hierarchical
                            Name auto-generated as A_<mode>_lr<lr>_ep<ep>_bs<bs>
                            Config and best_val_macro_f1 in summary panel

Reproducibility: set_seed(42) is called before data loading; identical
runs on the same hardware produce identical results.
"""

import argparse
import json
import os

from torch.utils.data import DataLoader, Subset
import wandb

from data import build_datasets, collate_fn, EMERGENCY_INTENTS
from models import build_model_a
from train import train
from utils import get_device, set_seed


def default_run_name(args: argparse.Namespace) -> str:
    """Auto-generate a W&B run name from condition + hyperparameters.

    Format: A_<mode>_lr<lr>_ep<epochs>_bs<batch_size>

    ``mode`` is "smoke" when either --limit_train or --limit_val is set
    (a reduced-size debug run), otherwise "full". This makes every run
    scannable in the W&B runs table without opening each one.

    Examples:
        A_full_lr3e-05_ep20_bs8   (converged 20-epoch baseline)
        A_full_lr1e-04_ep10_bs8   (the lr=1e-4 collapse, kept for record)
        A_smoke_lr3e-05_ep1_bs8   (16-sample smoke test)
    """
    mode = (
        "smoke"
        if (args.limit_train is not None or args.limit_val is not None)
        else "full"
    )
    return f"A_{mode}_lr{args.lr:.0e}_ep{args.epochs}_bs{args.batch_size}"


def main():
    parser = argparse.ArgumentParser(
        description="Condition A baseline: full fine-tuning of WavLM-base-plus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs. At 3e-5 the model is still improving at "
             "epoch 10; use 20 for a converged run.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Per-step batch size. 8 fits a T4 with 15 s audio.",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-5,
        help="Learning rate. 1e-4 causes collapse (constant predictions, "
             "loss frozen at ln 8); 1e-5 undertrains.",
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
        "--ckpt", type=str, default="checkpoints/A_best.pt",
        help="Path for the best checkpoint. Point at Google Drive for "
             "persistence across Colab sessions.",
    )
    parser.add_argument(
        "--save_config", action="store_true",
        help="Write a JSON config alongside the checkpoint "
             "(written pre-training and enriched post-training).",
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Override the auto-generated W&B run name.",
    )
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    run_name = args.run_name or default_run_name(args)
    print(f"Device: {device}")
    print(f"Run name: {run_name}")

    # ------------------------------------------------------------------
    # Data — build splits and derive emergency indices
    # ------------------------------------------------------------------
    splits, label_names, label2id = build_datasets()

    # Derive emergency ids from label *names* rather than hardcoding
    # [0, 3, 6, 7]. If a future dataset reorders labels, this stays correct.
    emergency_indices = sorted(
        label2id[n] for n in EMERGENCY_INTENTS if n in label2id
    )
    print(f"Labels: {label_names}")
    print(f"Emergency ids: {emergency_indices}")

    # Optional smoke-test subsets. Subset wraps the dataset lazily —
    # no audio is decoded until DataLoader calls __getitem__.
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

    # shuffle=True for training (breaks sample-order correlations);
    # shuffle=False for validation (deterministic, reproducible metrics).
    # num_workers=0 avoids Windows/Colab multiprocessing issues with the
    # HF dataset backend — safe default for this project's scale.
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

    # ------------------------------------------------------------------
    # Model — WavLM-base-plus with a fresh 8-class head
    # ------------------------------------------------------------------
    # Encoder weights load from the Hub; the classification head is
    # randomly initialized. transformers will emit a "newly initialized"
    # warning for the head only — this is expected and not an error.
    model = build_model_a(num_labels=len(label_names))

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project="taigispeech-hierarchical",
            name=run_name,
            config={
                "condition": "A",
                "backbone": "microsoft/wavlm-base-plus",
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "run_name": run_name,
            },
        )

    # ------------------------------------------------------------------
    # Checkpoint directory and config snapshot
    # ------------------------------------------------------------------
    # Guard against a bare filename like --ckpt A_best.pt, where
    # os.path.dirname() returns "" and os.makedirs("") would raise.
    ckpt_dir = os.path.dirname(args.ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Config snapshot uses the same base path as the checkpoint, so wherever
    # the .pt lands the _config.json lands next to it. Passing a Drive path
    # to --ckpt thus auto-persists the config to Drive too, with no extra flag.
    config_path = (
        args.ckpt.replace(".pt", "_config.json") if args.save_config else None
    )
    config_snapshot = {
        "condition": "A",
        "run_name": run_name,
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

    # Write the pre-training config FIRST so that even if training crashes
    # mid-run, the intended hyperparameters are on disk for post-mortem.
    if config_path is not None:
        with open(config_path, "w") as f:
            json.dump(config_snapshot, f, indent=2)
        print(f"Config (pre-training) written to: {config_path}")

    # ------------------------------------------------------------------
    # Train (loop, checkpointing, early stopping live in train.py)
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
        # Summary fields appear in a fixed panel in the W&B UI and are
        # sortable/filterable in the runs table. Putting best-F1 and
        # best-epoch here makes the winning condition instantly visible
        # across all A–E runs.
        wandb_run.summary["best_val_macro_f1"] = best_f1
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()


if __name__ == "__main__":
    main()
