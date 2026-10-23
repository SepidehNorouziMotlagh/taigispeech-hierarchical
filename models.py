
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


def build_model_b(num_labels: int = 8,
                  lora_r: int = 8,
                  lora_alpha: int = 32,
                  target_modules: list | None = None,
                  lora_dropout: float = 0.05):
    """Condition B: WavLM-base-plus + LoRA adapters (backbone frozen).

    Implementation note — why ``task_type=None``:

        PEFT's ``SEQ_CLS`` wrapper is text-only and always forwards
        ``input_ids`` to the base model. WavLM consumes waveforms and its
        forward does not accept ``input_ids``, so ``SEQ_CLS`` raises
        ``TypeError: WavLMForSequenceClassification.forward() got an
        unexpected keyword argument 'input_ids'``. Setting
        ``task_type=None`` returns the generic ``PeftModel``, which
        forwards ``input_values`` intact.

        Because ``SEQ_CLS`` normally auto-configures the classification
        head as trainable, the head layers must now be listed explicitly
        in ``modules_to_save``; otherwise they stay frozen and the model
        cannot learn to classify.

    Args:
        num_labels:     number of intent classes (8).
        lora_r:         LoRA rank (low-rank bottleneck dimension).
        lora_alpha:     LoRA scaling factor; effective scale is alpha / r.
        target_modules: attention projection names to adapt. Defaults to
                        ``["q_proj", "v_proj"]``.
        lora_dropout:   dropout on the LoRA branch during training.

    Returns:
        PeftModel wrapping WavLMForSequenceClassification.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    base = WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,  # required for WavLM (see docstring)
        modules_to_save=["projector", "classifier"],
    )

    return get_peft_model(base, lora_config)


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
