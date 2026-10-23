"""Model builders for all five project conditions.

Condition A: full fine-tuning of WavLM-base-plus (all params trainable).
Condition B: WavLM-base-plus + LoRA adapters (backbone frozen).

Conditions C, D, E will be added in later phases.

All builders return a model whose forward signature accepts:

    forward(input_values, attention_mask=None, labels=None) -> outputs

with ``outputs.loss`` and ``outputs.logits`` populated, so that the shared
training loop in train.py works identically across all conditions.
"""

from transformers import WavLMForSequenceClassification
from peft import LoraConfig, get_peft_model

BACKBONE_NAME = "microsoft/wavlm-base-plus"


def build_model_a(num_labels: int = 8):
    """Condition A: full fine-tuning of WavLM-base-plus, single 8-class head.

    Every parameter is trainable. This is the reference baseline and
    reproduces the Scratch (C=8) column of Chang et al. (2025), Table 8.

    Args:
        num_labels: number of intent classes (8 for TaigiSpeech).

    Returns:
        WavLMForSequenceClassification with randomly initialized head.
    """
    return WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )


def build_model_b(num_labels: int = 8,
                  lora_r: int = 8,
                  lora_alpha: int = 32,
                  target_modules: list | None = None,
                  lora_dropout: float = 0.05):
    """Condition B: WavLM-base-plus + LoRA adapters (backbone frozen).

    Inject low-rank adapters into the attention projections of every
    transformer layer. The base WavLM weights are frozen; only the LoRA
    matrices and the classification head (``projector``, ``classifier``)
    receive gradients. Total trainable parameters are ~0.5% of the backbone.

    Implementation note — why ``task_type=None``:

        PEFT's ``SEQ_CLS`` wrapper (``PeftModelForSequenceClassification``)
        is designed for text models. Its forward signature is:

            forward(input_ids, attention_mask, ..., **kwargs)

        and it always passes ``input_ids=input_ids`` to the base model.
        WavLM consumes waveforms, not token ids, and its forward does NOT
        accept ``input_ids`` — so using ``SEQ_CLS`` raises:

            TypeError: WavLMForSequenceClassification.forward() got an
            unexpected keyword argument 'input_ids'

        Setting ``task_type=None`` returns the generic ``PeftModel``, whose
        forward is:

            forward(*args, **kwargs)

        and forwards ``input_values`` through unchanged. Because ``SEQ_CLS``
        normally auto-configures the head as trainable, we must now list the
        head layers explicitly in ``modules_to_save``.

    Args:
        num_labels:     number of intent classes (8).
        lora_r:         LoRA rank (dimension of the low-rank bottleneck).
        lora_alpha:     LoRA scaling factor; effective scale is alpha / r.
        target_modules: attention projection names to adapt. Defaults to
                        ``["q_proj", "v_proj"]`` — the minimum recommendation
                        from the LoRA paper for transformer attention.
        lora_dropout:   dropout on the LoRA branch during training.

    Returns:
        PeftModel wrapping WavLMForSequenceClassification. The wrapper
        exposes ``.state_dict()`` returning LoRA weights + classification
        head only (a few MB, not the 380 MB base backbone).
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
        # task_type=None is REQUIRED for audio models — see docstring.
        task_type=None,
        # Because task_type=None, PEFT does not know which layers form the
        # classification head. Listing them here keeps them trainable and
        # includes them in the adapter checkpoint. Names match the top-level
        # submodules of WavLMForSequenceClassification.
        modules_to_save=["projector", "classifier"],
    )

    return get_peft_model(base, lora_config)
