from guidance.llms import Transformers

# from transformers import LlamaForCausalLM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# from optimum.bettertransformer import BetterTransformer
import os
import sys
import glob
import torch

from model_roles import Llama2ChatRole, Llama2GuanacoRole

selected_role = None


def get_role():
    return selected_role
    # return Vicuna1_3Role
    # return Llama2ChatRole
    # return Llama2GuanacoRole
    # return Llama2UncensoredChatRole


class LLaMATransformer(Transformers):
    """A HuggingFace transformers version of the LLaMA language model with Guidance support."""

    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str with LLaMAAutoGPTQ"

        global selected_role
        if "guanaco" in model.lower():
            print("found a Guanaco model")
            selected_role = Llama2GuanacoRole
        elif "llama-2-7b-chat" in model.lower():
            print("found a llama2chat model")
            selected_role = Llama2ChatRole
        elif "llama-2-13b-chat" in model.lower():
            print("found a llama2chat model")
            selected_role = Llama2ChatRole

        print(f"Initializing LLaMAAutoGPTQ with model {model}")

        low_vram_mode = "--low-vram" in sys.argv
        if low_vram_mode:
            print("low vram mode enabled")

        tokenizer = AutoTokenizer.from_pretrained(model)

        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=low_vram_mode,
            bnb_4bit_quant_type="nf4",
        )

        print("loading config...")
        config = AutoConfig.from_pretrained(model)
        print("loaded.")
        config.pretraining_tp = 1
        model_file = model
        # model = AutoModelForCausalLM.from_config(config)
        # model = AutoModelForCausalLM.from_pretrained(
        #     model,
        #     config=config,
        #     quantization_config=double_quant_config,
        #     torch_dtype=torch.float16,
        #     load_in_4bit=True,
        #     device_map="auto",
        # )

        from accelerate import init_empty_weights

        print("init empty weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        print("done.")

        from accelerate.utils import BnbQuantizationConfig

        quantization_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        from accelerate.utils import load_and_quantize_model

        from accelerate import infer_auto_device_map

        device_map = infer_auto_device_map(
            model, max_memory={0: "9GiB", "cpu": "60GiB"}
        )

        print(f"using device map: {device_map}")

        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 0,
            "model.layers.2": 0,
            "model.layers.3": 0,
            "model.layers.4": 0,
            "model.layers.5": 0,
            "model.layers.6.self_attn": 0,
            "model.layers.6.mlp.gate_proj": 0,
            "model.layers.6.mlp.up_proj": 0,
            "model.layers.6.mlp.down_proj": 0,
            "model.layers.6.mlp.act_fn": 0,
            "model.layers.6.input_layernorm": 0,
            "model.layers.6.post_attention_layernorm": 0,
            "model.layers.7": 0,
            "model.layers.8": 0,
            "model.layers.9": 0,
            "model.layers.10": 0,
            "model.layers.11": 0,
            "model.layers.12": 0,
            "model.layers.13": 0,
            "model.layers.14": 0,
            "model.layers.15": 0,
            "model.layers.16": 0,
            "model.layers.17": 0,
            "model.layers.18": 0,
            "model.layers.19": 0,
            "model.layers.20": 0,
            "model.layers.21": 0,
            "model.layers.22": 0,
            "model.layers.23": 0,
            "model.layers.24": 0,
            "model.layers.25": 0,
            "model.layers.26": 0,
            "model.layers.27": 0,
            "model.layers.28": 0,
            "model.layers.29": 0,
            "model.layers.30": 0,
            "model.layers.31": 0,
            "model.layers.32": 0,
            "model.layers.33": 0,
            "model.layers.34": "cpu",
            "model.layers.35": "cpu",
            "model.layers.36": "cpu",
            "model.layers.37": "cpu",
            "model.layers.38": "cpu",
            "model.layers.39": "cpu",
            "model.norm": "cpu",
            "lm_head": "cpu",
        }

        print("loading quantized model...")
        quantized_model = load_and_quantize_model(
            model,
            weights_location="/home/andy/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-Chat-fp16/snapshots/5f442b4db86c66e93d6844975606ae88978b35d0",
            bnb_quantization_config=quantization_config,
            device_map=device_map,
            # max_memory={0: "4GiB", "cpu": "18GiB"},
        )
        print("done")

        # model.config.max_seq_len = 4096  # this is the one

        return super()._model_and_tokenizer(quantized_model, tokenizer, **kwargs)

    @staticmethod
    def role_start(role):
        return get_role().role_start(role)

    @staticmethod
    def role_end(role):
        return get_role().role_end(role)


def find_safetensor_filename(dir):
    # Make sure the directory path ends with '/'
    if dir[-1] != "/":
        dir += "/"

    # Use glob to find all files with the given extension
    files = glob.glob(dir + "*.safetensors")

    # If there is at least one file, return the first one
    if len(files) == 0:
        print(f"Error: no safetensor file found in {dir}")
        return None
    elif len(files) == 1:
        return os.path.basename(files[0])
    else:
        print(f"Warning: multiple safetensor files found in {dir}, picking just one")
        return os.path.basename(files[0])
