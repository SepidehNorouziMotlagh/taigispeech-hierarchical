
"""
Training utilities for TaigiSpeech intent classification.

The training loop selects the best model using validation macro-F1.
Whenever that score improves, it saves the model's state_dict locally
and, when a W&B run is active, logs a versioned model artifact.

Checkpoint files contain model weights only. They are intended for
evaluation and inference, not exact resumption of optimizer state.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Train for one epoch and return the sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["intent"].to(device)

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    if n_samples == 0:
        raise ValueError("The training loader contains no samples.")

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, device, emergency_indices):
    """Evaluate intent classification and binary emergency detection."""
    model.eval()
    predictions_all = []
    labels_all = []
    emergency_all = []
    total_loss = 0.0
    n_samples = 0
    emergency_set = set(emergency_indices)

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["intent"].to(device)
        emergency = batch["emergency"].to(device)

        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )

        batch_size = labels.size(0)
        total_loss += outputs.loss.item() * batch_size
        n_samples += batch_size

        predictions_all.append(outputs.logits.argmax(dim=-1).cpu())
        labels_all.append(labels.cpu())
        emergency_all.append(emergency.cpu())

    if n_samples == 0:
        raise ValueError("The validation loader contains no samples.")

    predictions = torch.cat(predictions_all).numpy()
    labels = torch.cat(labels_all).numpy()
    emergency_labels = torch.cat(emergency_all).numpy()

    predicted_emergency = np.array(
        [int(prediction in emergency_set) for prediction in predictions]
    )

    return {
        "loss": total_loss / n_samples,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "emergency_recall": recall_score(
            emergency_labels, predicted_emergency, zero_division=0
        ),
        "emergency_f1": f1_score(
            emergency_labels,
            predicted_emergency,
            average="binary",
            zero_division=0,
        ),
    }


def save_best_checkpoint(
    model,
    checkpoint_path,
    wandb_run=None,
    artifact_name="taigispeech-condition-a",
    epoch_index=None,
    metrics=None,
):
    """
    Save model weights locally and optionally log a W&B model artifact.

    Local saving happens first, so a W&B upload problem does not prevent
    the best checkpoint from being saved in the current runtime.

    Args:
        model: Trained PyTorch model.
        checkpoint_path: Destination path for the local state_dict.
        wandb_run: Active W&B run, or None to disable artifact logging.
        artifact_name: Stable W&B artifact name; improvements create
            successive versions under this name.
        epoch_index: Zero-based epoch index, as used internally.
        metrics: Validation metrics associated with this checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Save weights first. This remains available even if cloud upload fails.
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[checkpoint] Saved local weights: {checkpoint_path}")

    if wandb_run is None:
        return

    metrics = metrics or {}
    human_epoch = epoch_index + 1 if epoch_index is not None else None

    metadata = {
        "condition": "A",
        "checkpoint_format": "pytorch_state_dict",
        "epoch_index_zero_based": epoch_index,
        "epoch_number_one_based": human_epoch,
        "selection_metric": "val_macro_f1",
        "val_macro_f1": metrics.get("macro_f1"),
        "val_loss": metrics.get("loss"),
        "val_accuracy": metrics.get("accuracy"),
        "val_emergency_recall": metrics.get("emergency_recall"),
        "val_emergency_f1": metrics.get("emergency_f1"),
    }

    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=(
                "Best validation-macro-F1 checkpoint for "
                "TaigiSpeech Condition A: full WavLM fine-tuning."
            ),
            metadata=metadata,
        )
        artifact.add_file(str(checkpoint_path), name="A_best.pt")

        logged_artifact = wandb_run.log_artifact(
            artifact,
            aliases=["best", "latest"],
        )

        # Wait for the upload to finish before training continues.
        logged_artifact.wait()

        print(
            "[checkpoint] W&B artifact uploaded: "
            f"{artifact_name} | epoch {human_epoch} | "
            f"val_macro_f1={metrics.get('macro_f1', float('nan')):.4f}"
        )

    except Exception as exc:
        # Do not discard a successful local save because the network,
        # W&B service, or artifact upload failed.
        print(
            "[checkpoint] WARNING: W&B artifact upload failed. "
            f"The local checkpoint remains at {checkpoint_path}. "
            f"Error: {exc}"
        )


def train(
    model,
    train_loader,
    val_loader,
    device,
    emergency_indices,
    epochs=10,
    lr=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=3,
    ckpt_path="checkpoints/A_best.pt",
    wandb_run=None,
    artifact_name="taigispeech-condition-a",
):
    """
    Train a model with early stopping and best-checkpoint tracking.

    The model is selected using validation macro-F1. Every improvement
    triggers a local checkpoint save and, if enabled, a W&B artifact
    upload. Epoch indices in W&B metrics are zero-based; artifact
    metadata also stores the corresponding one-based epoch number.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    if len(train_loader) == 0:
        raise ValueError("The training loader contains no batches.")
    if len(val_loader) == 0:
        raise ValueError("The validation loader contains no batches.")

    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    print(
        f"[train] total_steps={total_steps} | "
        f"batches/epoch={len(train_loader)} | "
        f"epochs={epochs} | patience={patience}"
    )

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        train_seconds = time.time() - epoch_start

        validation_start = time.time()
        val = evaluate(model, val_loader, device, emergency_indices)
        validation_seconds = time.time() - validation_start

        improved = val["macro_f1"] > best_f1

        if improved:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            no_improve = 0

            save_best_checkpoint(
                model=model,
                checkpoint_path=ckpt_path,
                wandb_run=wandb_run,
                artifact_name=artifact_name,
                epoch_index=epoch,
                metrics=val,
            )
        else:
            no_improve += 1

        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"val_acc={val['accuracy']:.4f} | "
            f"val_macro_f1={val['macro_f1']:.4f} | "
            f"val_emg_recall={val['emergency_recall']:.4f} | "
            f"train_t={train_seconds:.1f}s "
            f"val_t={validation_seconds:.1f}s"
            + (" *NEW BEST*" if improved else "")
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "epoch_number": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                    "val_emergency_recall": val["emergency_recall"],
                    "val_emergency_f1": val["emergency_f1"],
                    "best_val_macro_f1": best_f1,
                    "best_epoch_number": best_epoch + 1,
                    "no_improve": no_improve,
                }
            )

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement for {patience} epochs)."
            )
            break

    print(
        f"Best epoch: {best_epoch + 1} | "
        f"best val macro-F1: {best_f1:.4f}"
    )

    return best_epoch, best_f1
