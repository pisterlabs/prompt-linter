import guidance
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common_lit_kaggle.settings.config import Config


def load_llm():
    # pylint: disable=no-member
    config = Config.get()
    model = config.zero_shot_model

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
        device_map={"": 0},
    )
    print(type(model))

    return guidance.llms.Transformers(model=model, tokenizer=tokenizer)
