"""
Model definitions for TaigiSpeech hierarchical intent recognition.

Condition A:
    Standard WavLM-base-plus with full fine-tuning.

Condition B:
    WavLM-base-plus with LoRA adapters on q_proj and v_proj.
    WavLM's attention implementation uses projection weights directly
    inside torch.nn.functional.multi_head_attention_forward, bypassing
    the projection modules' forward() methods. The attention patch below
    explicitly incorporates the LoRA weight updates so they receive gradients.

Important:
    This implementation uses lora_dropout=0.0 because it computes the
    effective LoRA weights directly. A nonzero LoRA dropout cannot be
    faithfully represented by a single effective weight matrix.
"""

import types

import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers import WavLMForSequenceClassification
from transformers.models.wavlm.modeling_wavlm import WavLMAttention

from data import BACKBONE_NAME


# ---------------------------------------------------------------------
# Condition A: full fine-tuning
# ---------------------------------------------------------------------

def build_model_a(num_labels=8):
    """
    Build the standard WavLM sequence-classification model.

    All model parameters are trainable.
    """
    model = WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )

    return model


# ---------------------------------------------------------------------
# Shared LoRA attention helpers
# ---------------------------------------------------------------------

def _effective_lora_weight(module):
    """
    Return a linear layer's effective weight.

    For an ordinary nn.Linear layer, return its existing weight.

    For a PEFT LoRA-wrapped linear layer, return:

        W_effective = W_base + scaling * (B @ A)

    The calculation remains differentiable, allowing gradients to flow
    into the LoRA A and B matrices.

    This function assumes standard LoRA without DoRA or other adapter
    methods that change the effective-weight calculation.
    """
    # Unwrapped layers, such as WavLM's k_proj.
    if not hasattr(module, "get_base_layer"):
        return module.weight

    base_layer = module.get_base_layer()
    weight = base_layer.weight

    # A PEFT-wrapped layer without LoRA matrices behaves as its base layer.
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return weight

    active_adapters = module.active_adapters

    # PEFT versions may expose active_adapters as a string or a list.
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]

    for adapter_name in active_adapters:
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
    attention_mask=None,
    gated_position_bias=None,
    output_attentions=False,
):
    """
    Replacement forward method for WavLMAttention.

    Unlike the original WavLM implementation, this method passes
    effective Q/K/V weights into multi_head_attention_forward, ensuring
    that the LoRA updates on q_proj and v_proj are actually used.
    """
    # WavLM uses sequence-first layout inside multi_head_attention_forward.
    query = key = value = hidden_states.transpose(0, 1)

    key_padding_mask = (
        attention_mask.ne(1)
        if attention_mask is not None
        else None
    )

    # Obtain base weights plus LoRA updates where applicable.
    q_weight = _effective_lora_weight(self.q_proj)
    k_weight = _effective_lora_weight(self.k_proj)
    v_weight = _effective_lora_weight(self.v_proj)

    # WavLM stores Q/K/V biases separately but supplies them together
    # to PyTorch's multi-head attention implementation.
    in_proj_bias = torch.cat(
        (
            self.q_proj.bias,
            self.k_proj.bias,
            self.v_proj.bias,
        ),
        dim=0,
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

    # Return to WavLM's expected batch-first layout.
    attn_output = attn_output.transpose(0, 1)

    if attn_weights is not None:
        attn_weights = attn_weights[:, None].broadcast_to(
            attn_weights.shape[:1]
            + (self.num_heads,)
            + attn_weights.shape[1:]
        )

    return attn_output, attn_weights


def enable_wavlm_lora_attention(model):
    """
    Patch each WavLM attention module to use effective LoRA weights.

    Call this after applying PEFT and before training.
    """
    patched_count = 0

    for module in model.modules():
        if isinstance(module, WavLMAttention):
            module.forward = types.MethodType(
                _wavlm_lora_attention_forward,
                module,
            )
            patched_count += 1

    if patched_count == 0:
        raise RuntimeError(
            "No WavLMAttention modules were found. "
            "Check the Transformers model architecture/version."
        )

    print(
        "LoRA-compatible WavLM attention enabled: "
        f"{patched_count} layers."
    )

    return model


# ---------------------------------------------------------------------
# Condition B: LoRA fine-tuning
# ---------------------------------------------------------------------

def build_model_b(
    num_labels=8,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.0,
):
    """
    Build WavLM with LoRA adapters for Condition B.

    LoRA is applied to q_proj and v_proj. The classification projector
    and classifier remain trainable through modules_to_save.

    lora_dropout must be 0.0 for this effective-weight implementation.
    """
    if lora_dropout != 0.0:
        raise ValueError(
            "This WavLM attention patch requires lora_dropout=0.0. "
            "Set --lora_dropout 0.0 when running Condition B."
        )

    model = WavLMForSequenceClassification.from_pretrained(
        BACKBONE_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        task_type=None,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["projector", "classifier"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Essential: standard WavLM attention bypasses q_proj/v_proj forward().
    model = enable_wavlm_lora_attention(model)

    # Report trainable parameter count for experiment logging.
    model.print_trainable_parameters()

    return model

def count_parameters(model):
    """
    Return total and trainable parameter counts.
    """
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ---------------------------------------------------------------------
# Quick local verification
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Building Condition B for a forward-pass smoke test...")

    model = build_model_b()
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Short synthetic waveform. This checks execution, not model quality.
    input_values = torch.randn(1, 16000, device=device)
    attention_mask = torch.ones_like(
        input_values,
        dtype=torch.long,
    )

    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
        )

    print("Device:", device)
    print("Logits shape:", tuple(outputs.logits.shape))
    print("Forward-pass smoke test completed.")
