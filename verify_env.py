import platform
import sys

import datasets
import peft
import sklearn
import torch
import transformers
import wandb

import utils


def main() -> None:
    print(f"Python: {sys.version.split()[0]} on {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"datasets: {datasets.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"sklearn: {sklearn.__version__}")
    print(f"wandb: {wandb.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    utils.set_seed(42)
    print("Seed set to 42.")

    try:
        from datasets import load_dataset_builder
        load_dataset_builder("TaigiSpeech/TaigiSpeech")
        print("TaigiSpeech access OK.")
    except Exception as e:
        print(f"TaigiSpeech access FAILED: {e}")
        print("If gated, run: huggingface-cli login")

    run = wandb.init(
        project="taigispeech-hierarchical",
        name="env-check",
        mode="online",
    )
    wandb.log({"env_ok": 1})
    run.finish()
    print("W&B logging OK.")


if __name__ == "__main__":
    main()