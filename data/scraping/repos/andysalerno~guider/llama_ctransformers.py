from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from guidance.llms import Transformers

import os
import sys
import glob
import torch

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

        print(f"Initializing ctransformers with model {model}")

        tokenizer = AutoTokenizer.from_pretrained("TheBloke/StableBeluga-13B-GPTQ")

        model = AutoModelForCausalLM.from_pretrained(model, model_type="llama")
        model.device = "cpu"

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


class Vicuna1_3Role:
    @staticmethod
    def role_start(role):
        if role == "user":
            return "USER: "
        elif role == "assistant":
            return "ASSISTANT: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""


class Llama2GuanacoRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            return "### Human: "
        elif role == "assistant":
            return "### Assistant: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            # return ""
            return "</s>"
        else:
            return ""


class RedmondPufferRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            return "USER: "
        elif role == "assistant":
            return "ASSISTANT: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""


class Llama2ChatRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            # return " <s>[INST] "
            # return " [INST] "
            return ""
        elif role == "assistant":
            return ""
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return " [/INST] "
        elif role == "assistant":
            # return " </s>"
            # return ""
            return " </s><s>[INST] "
        else:
            return ""


class Llama2UncensoredChatRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            # return " <s>[INST] "
            return "### HUMAN:\n"
        elif role == "assistant":
            return "### RESPONSE:\n"
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return "\n"
        elif role == "assistant":
            # return " </s>"
            return "\n"
        else:
            return ""
