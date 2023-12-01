from guidance.llms import Transformers
from huggingface_hub import snapshot_download
import os
import glob
from transformers import AutoTokenizer
from model_roles import get_role_from_model_name
from awq import AutoAWQForCausalLM
from pathlib import Path


selected_role = None


def get_role():
    return selected_role


class LlamaOpenAIAPI(Transformers):
    """An AWK version of the LLaMA language model with Guidance support."""

    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str"

        global selected_role
        selected_role = get_role_from_model_name(model)

        print(f"Initializing LLaMAAwk with model {model}")

        branch = None
        if ":" in model:
            splits = model.split(":")
            model = splits[0]
            branch = splits[1]

        models_dir = "./models"
        name_suffix = model.split("/")[1]
        model_dir = f"{models_dir}/{name_suffix}"

        print(f'invoking: {model}, {branch}, {model_dir}')
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model, revision=branch, local_dir=model_dir)

        model_basename = find_safetensor_filename(model_dir)

        print(f"found model with basename {model_basename} in dir {model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        model = AutoAWQForCausalLM.from_quantized(model_dir, model_basename, fuse_layers=True)

        model.device = "0"

        # model.config.max_seq_len = 4096  # this is the one

        return super()._model_and_tokenizer(model, tokenizer, **kwargs)

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
