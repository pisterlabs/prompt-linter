import os

import torch
import transformers
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

MODEL_IDS = {
    "llama2_chat": "meta-llama/Llama-2-7b-chat-hf",
    "llava": "liuhaotian/llava-v1.5-7b",
}


def get_llama2_chat():
    assert (
        "HF_AUTH_TOKEN" in os.environ
    ), "Must set the HF_AUTH_TOKEN environment variable."

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_IDS["llama2_chat"],
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto",
        use_auth_token=os.environ["HF_AUTH_TOKEN"],
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_IDS["llama2_chat"], use_auth_token=os.environ["HF_AUTH_TOKEN"]
    )

    # TODO: add stopping criteria

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        temperature=0.15,
        max_new_tokens=512,
        repetition_penalty=1.2,
    )

    llm = HuggingFacePipeline(pipeline=pipeline)

    return llm


def get_llava_chat():
    class LlavaConfig(transformers.LlamaConfig):
        model_type = "llava"

    transformers.AutoConfig.register("llava", LlavaConfig)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LlavaLlamaForCausalLM.from_pretrained(
        MODEL_IDS["llava"],
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map="auto",
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_IDS["llava"])

    # TODO: add stopping criteria

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        temperature=0.15,
        max_new_tokens=512,
        repetition_penalty=1.2,
    )

    llm = HuggingFacePipeline(pipeline=pipeline)

    return llm
