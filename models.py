"""
Model definitions for TaigiSpeech intent classification.

Condition A: WavLM-base-plus full fine-tuning.
Condition B: WavLM-base-plus with LoRA-adjusted Q/V projections.

Compatibility target:
    transformers==4.57.6
"""

import types

import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers import WavLMForSequenceClassification
from transformers.models.wavlm.modeling_wavlm import WavLMAttention


BACKBONE_NAME = "microsoft/wavlm-base-plus"


def build_model_a(num_labels=8):
    """Build the full fine-tuning baseline."""
    model = WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    return model


def _effective_lora_weight(module):
    """
    Compute W_eff = W_base + scaling * (B @ A).

    This implementation is for ordinary LoRA with bias='none',
    without DoRA or other weight parametrizations.
    """
    base_layer = module.get_base_layer()
    weight = base_layer.weight

    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return weight

    for adapter_name in module.active_adapters:
        if adapter_name not in module.lora_A:
            continue
        if adapter_name not in module.lora_B:
            continue

        A = module.lora_A[adapter_name].weight
        B = module.lora_B[adapter_name].weight
        scaling = module.scaling[adapter_name]

        weight = weight + scaling * (B @ A)

    return weight


def _wavlm_lora_attention_forward(
    self,
    hidden_states,
    attention_mask,
    gated_position_bias,
    output_attentions,
):
    """
    WavLM attention forward pass with differentiable LoRA-adjusted
    query and value weights.

    LoRA dropout must be 0 because this implementation represents
    the LoRA update as an effective weight matrix.
    """
    query = key = value = hidden_states.transpose(0, 1)

    key_padding_mask = (
        attention_mask.ne(1)
        if attention_mask is not None
        else None
    )

    q_weight = _effective_lora_weight(self.q_proj)
    k_weight = _effective_lora_weight(self.k_proj)
    v_weight = _effective_lora_weight(self.v_proj)

    in_proj_bias = torch.cat(
        [
            self.q_proj.bias,
            self.k_proj.bias,
            self.v_proj.bias,
        ]
    )

    attn_output, attn_weights = F.multi_head_attention_forward(
        query=query,
        key=key,
        value=value,
        embed_dim_to_check=self.embed_dim,
        num_heads=self.num_heads,
        in_proj_weight=torch.empty(
            0,
            device=q_weight.device,
            dtype=q_weight.dtype,
        ),
        in_proj_bias=in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=self.dropout,
        out_proj_weight=self.out_proj.weight,
        out_proj_bias=self.out_proj.bias,
        training=self.training,
        key_padding_mask=key_padding_mask,
        need_weights=output_attentions,
        attn_mask=gated_position_bias,
        use_separate_proj_weight=True,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
    )

    attn_output = attn_output.transpose(0, 1)

    if attn_weights is not None:
        attn_weights = attn_weights[:, None].broadcast_to(
            attn_weights.shape[:1]
            + (self.num_heads,)
            + attn_weights.shape[1:]
        )

    return attn_output, attn_weights


def enable_wavlm_lora_attention(model):
    """Patch every WavLM attention layer to use effective LoRA weights."""
    count = 0

    for module in model.modules():
        if isinstance(module, WavLMAttention):
            module.torch_multi_head_self_attention = types.MethodType(
                _wavlm_lora_attention_forward,
                module,
            )
            count += 1

    if count != 12:
        raise RuntimeError(
            f"Expected 12 WavLM attention layers, found {count}."
        )

    print(f"LoRA-compatible WavLM attention enabled: {count} layers.")
    return model


def build_model_b(
    num_labels=8,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
):
    """Build the LoRA model with differentiable Q/V updates."""

    if lora_dropout != 0.0:
        raise ValueError(
            "This weight-based WavLM attention patch requires "
            "lora_dropout=0.0. Set it explicitly."
        )

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
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["projector", "classifier"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model = enable_wavlm_lora_attention(model)

    return model


def count_parameters(model):
    """Count total and trainable model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": (
            100.0 * trainable / total if total else 0.0
        ),
    }
