import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, n = 0.0, 0
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

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, emergency_indices):
    model.eval()
    preds_all, labels_all, emg_all = [], [], []
    total_loss, n = 0.0, 0
    emg_set = set(emergency_indices)

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
        total_loss += outputs.loss.item() * labels.size(0)
        n += labels.size(0)

        preds = outputs.logits.argmax(dim=-1)
        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())
        emg_all.append(emergency.cpu())

    preds_np = torch.cat(preds_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    emg_np = torch.cat(emg_all).numpy()
    pred_emg = np.array([1 if p in emg_set else 0 for p in preds_np])

    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(labels_np, preds_np),
        "macro_f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "emergency_recall": recall_score(emg_np, pred_emg, zero_division=0),
        "emergency_f1": f1_score(emg_np, pred_emg, average="binary", zero_division=0),
    }


def train(model, train_loader, val_loader, device, emergency_indices,
          epochs=10, lr=1e-5, weight_decay=0.01, warmup_ratio=0.1,
          patience=3, ckpt_path="checkpoints/A_best.pt", wandb_run=None):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    print(f"[train] total_steps={total_steps} | "
          f"batches/epoch={len(train_loader)} | epochs={epochs} | patience={patience}")

    best_f1, best_epoch, no_improve = -1.0, -1, 0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        t1 = time.time()
        val = evaluate(model, val_loader, device, emergency_indices)
        t2 = time.time()

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        print(
            f"epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | val_acc={val['accuracy']:.4f} | "
            f"val_macro_f1={val['macro_f1']:.4f} | "
            f"val_emg_recall={val['emergency_recall']:.4f} | "
            f"train_t={t1-t0:.1f}s val_t={t2-t1:.1f}s" + (" *NEW*" if improved else "")
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "val_emergency_recall": val["emergency_recall"],
                "val_emergency_f1": val["emergency_f1"],
                "best_val_macro_f1": best_f1,
                "no_improve": no_improve,
            })

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs).")
            break

    print(f"Best epoch: {best_epoch+1} | best val macro-F1: {best_f1:.4f}")
    return best_epoch, best_f1