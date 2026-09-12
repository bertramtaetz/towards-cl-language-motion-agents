import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


def test_merged_lora_adapter_creation_and_activation():
    """Smoke-test that our merged-adapter helper creates an adapter and can activate it.

    This does not depend on MotionLLM or large models.
    """

    from continual_learning.olora_moe.topk_token_inference import (
        MergedAdapterSpec,
        ensure_merged_lora_adapter,
    )

    base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    cfg = LoraConfig(
        r=2,
        lora_alpha=2,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    m = get_peft_model(base, cfg, adapter_name="task_0")
    m.add_adapter("task_1", cfg)

    # Make adapters non-trivial.
    with torch.no_grad():
        for _, module in m.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                if "task_0" in getattr(module, "lora_A"):
                    module.lora_A["task_0"].weight.fill_(0.1)
                    module.lora_B["task_0"].weight.fill_(0.2)
                if "task_1" in getattr(module, "lora_A"):
                    module.lora_A["task_1"].weight.fill_(0.3)
                    module.lora_B["task_1"].weight.fill_(0.4)

    merged = ensure_merged_lora_adapter(
        peft_model=m,
        spec=MergedAdapterSpec(
            merged_adapter_name="merged_0_1",
            source_adapter_names=("task_0", "task_1"),
        ),
        base_lora_config=m.peft_config["task_0"],
    )
    assert merged == "merged_0_1"
    assert merged in m.peft_config

    # Ensure we can activate the merged adapter.
    m.set_adapter(merged)
