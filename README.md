# TaigiSpeech Hierarchical Intent Recognition

Does explicitly modeling the emergency hierarchy improve fine-grained intent
recognition in low-resource speech?

Backbone: `microsoft/wavlm-base-plus` + LoRA.
Dataset: [TaigiSpeech](https://huggingface.co/datasets/TaigiSpeech/TaigiSpeech).

## Setup

    pip install -r requirements.txt
    wandb login
    python -m taigi.verify_env

## Layout

All code lives in the `taigi/` package. Run any script from the project root
as `python -m taigi.<script>`.