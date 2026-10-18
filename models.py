from transformers import WavLMForSequenceClassification
BACKBONE_NAME='microsoft/wavlm-base-plus'

def build_model_a(num_labels=8):
    """Condition A: full fine-tuning of WavLM-base-plus, single 8-class head."""
    model=WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type='single_label_classification',
        ignore_mismatched_sizes=True,
    )
    return model