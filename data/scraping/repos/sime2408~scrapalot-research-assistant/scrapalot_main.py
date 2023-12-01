#!/usr/bin/env python3
import logging
import os

import torch
from auto_gptq import AutoGPTQForCausalLM
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain import HuggingFacePipeline
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import LlamaCpp, GPT4All, OpenAI
from torch import cuda as torch_cuda
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline

from scripts.app_environment import model_type, openai_api_key, model_n_ctx, model_temperature, model_top_p, model_n_batch, model_use_mlock, model_verbose, \
    args, gpt4all_backend, model_path_or_id, gpu_is_enabled, cpu_model_n_threads, gpu_model_n_threads, huggingface_model_base_name

# Ensure TOKENIZERS_PARALLELISM is set before importing any HuggingFace module.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# load environment variables

try:
    load_dotenv()
except Exception as e:
    logging.error("Error loading .env file, create one from example.env:", str(e))


def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0] / (1024 ** 2))


# noinspection PyPep8Naming
def calculate_layer_count() -> None | int | float:
    """
    How many layers of a neural network model you can fit into the GPU memory,
    rather than determining the number of threads.
    The layer size is specified as a constant (120.6 MB), and the available GPU memory is divided by this to determine the maximum number of layers that can be fit onto the GPU.
    Some additional memory (the size of 6 layers) is reserved for other uses.
    The maximum layer count is capped at 43.
    """
    if not gpu_is_enabled:
        return None
    LAYER_SIZE_MB = 120.6  # This is the size of a single layer on VRAM, and is an approximation.
    # The current set value is for 7B models. For other models, this value should be changed.
    LAYERS_TO_REDUCE = 6  # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
    if (get_gpu_memory() // LAYER_SIZE_MB) - LAYERS_TO_REDUCE >= 43:
        return 43
    else:
        return get_gpu_memory() // LAYER_SIZE_MB - LAYERS_TO_REDUCE


def get_llm_instance(*callback_handler: BaseCallbackHandler):
    logging.debug(f"Initializing model...")

    callbacks = [] if args.mute_stream else callback_handler

    if model_type == "gpt4all":
        if gpu_is_enabled:
            logging.warn("GPU is enabled, but GPT4All does not support GPU acceleration. Please use LlamaCpp instead.")
            exit(1)
        return GPT4All(
            model=model_path_or_id,
            n_ctx=model_n_ctx,
            max_tokens=model_n_ctx,
            backend=gpt4all_backend,
            callbacks=callbacks,
            use_mlock=model_use_mlock,
            n_threads=gpu_model_n_threads if gpu_is_enabled else cpu_model_n_threads,
            n_predict=1000,
            n_batch=model_n_batch,
            top_p=model_top_p,
            temp=model_temperature,
            streaming=True,
            verbose=False
        )
    elif model_type == "llamacpp":
        return LlamaCpp(
            model_path=model_path_or_id,
            temperature=model_temperature,
            n_ctx=model_n_ctx,
            max_tokens=model_n_ctx,
            top_p=model_top_p,
            n_batch=model_n_batch,
            use_mlock=model_use_mlock,
            n_threads=gpu_model_n_threads if gpu_is_enabled else cpu_model_n_threads,
            verbose=model_verbose,
            n_gpu_layers=calculate_layer_count() if gpu_is_enabled else None,
            callbacks=callbacks,
            streaming=True
        )
    elif model_type == "huggingface":
        if huggingface_model_base_name is not None:
            if not gpu_is_enabled:
                logging.info("Using Llamacpp for quantized models")
                model_path = hf_hub_download(local_dir=os.path.abspath('models'), local_dir_use_symlinks=True, repo_id=model_path_or_id, filename=huggingface_model_base_name)
                return LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, max_tokens=model_n_ctx, temperature=model_temperature, repeat_penalty=1.15)

            else:
                logging.info("Using AutoGPTQForCausalLM for quantized models")
                tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, use_fast=True)
                logging.info("Tokenizer loaded")

                model = AutoGPTQForCausalLM.from_quantized(
                    model_name_or_path=model_path_or_id,
                    model_basename=huggingface_model_base_name if ".safetensors" not in huggingface_model_base_name else huggingface_model_base_name.replace(".safetensors", ""),
                    use_safetensors=True,
                    trust_remote_code=True,
                    device="cuda:0",
                    use_triton=False,
                    quantize_config=None,
                )
        else:
            if gpu_is_enabled:
                logging.info("Using AutoModelForCausalLM for full models")
                tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path_or_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    # max_memory={0: "15GB"} # Uncomment this line if you encounter CUDA out of memory errors
                )
                model.tie_weights()
            else:
                logging.info("Using LlamaTokenizer")
                tokenizer = LlamaTokenizer.from_pretrained(model_path_or_id)
                model = LlamaForCausalLM.from_pretrained(model_path_or_id)

        return HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=model_n_ctx,
            max_new_tokens=model_n_ctx,
            temperature=model_temperature,
            top_p=model_top_p,
            repetition_penalty=1.15,
            generation_config=GenerationConfig.from_pretrained(model_path_or_id),
        ))
    elif model_type == "openai":
        assert openai_api_key is not None, "Set ENV OPENAI_API_KEY, Get one here: https://platform.openai.com/account/api-keys"
        return OpenAI(openai_api_key=openai_api_key, callbacks=callbacks)
    else:
        logging.error(f"Model {model_type} not supported!")
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
