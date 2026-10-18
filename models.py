
"""Model builders for TaigiSpeech intent classification."""

from peft import LoraConfig, TaskType, get_peft_model
from transformers import WavLMForSequenceClassification

BACKBONE_NAME = "microsoft/wavlm-base-plus"


def build_model_a(num_labels=8):
    """Condition A: full fine-tuning with a single intent-classification head."""
    model = WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    return model


def build_model_b(
    num_labels=8,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
):
    """
    Condition B: LoRA fine-tuning of WavLM-base-plus.

    LoRA is applied to query and value projections in the attention layers.
    The task-specific projector and classifier remain trainable.
    """
    base_model = WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        task_type=None,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["projector", "classifier"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    return model


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
    }
