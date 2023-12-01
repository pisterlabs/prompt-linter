from asyncio.log import logger
from functools import partial
import os

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoConfig,
)
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map,
)
import requests
import openai
import torch
import gc
import cohere
import time
import re

MODEL_SETTINGS = {
    "google/flan-t5-xxl": {
        "checkpoint": "flan-t5-xxl",
        "no_split_module_classes": ["T5Block"],
    },
    "EleutherAI/gpt-j-6B": {
        "checkpoint": "sharded-gpt-j-6B",
        "no_split_module_classes": ["GPTJBlock"],
    },
    "facebook/opt-30b": {
        "checkpoint": "opt-30b",
        "no_split_module_classes": ["OPTDecoderLayer"],
    },
    "facebook/opt-13b": {
        "checkpoint": "opt-13b",
        "no_split_module_classes": ["OPTDecoderLayer"],
    },
    "facebook/opt-125m": {
        "checkpoint": "opt-125m",
        "no_split_module_classes": ["OPTDecoderLayer"],
    },
    "t5-11b": {"checkpoint": "t5-11b", "no_split_module_classes": ["T5Block"]},
    "ul2": {"checkpoint": "ul2", "no_split_module_classes": ["T5Block"]},
}


def gpt3(prompt, max_tokens=3, model_name="text-ada-001"):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        max_tokens=max_tokens,
        top_p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"]


def hf_inference_api(prompt, model_name, max_tokens=3):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"

    query_input = {
        "inputs": prompt,
    }
    if model_name not in ["bigscience/T0pp"]:
        query_input["parameters"] = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "top_p": 0.9,
            "temperature": 0.1,
        }
    headers = {"Authorization": f"Bearer {os.environ.get('HF_API_KEY')}"}

    response = requests.post(API_URL, headers=headers, json=query_input)
    output = response.json()
    if type(output) != list:
        logger.error("Bad response from HF.")
        logger.error(output)
        return False

    output = output[0]["generated_text"]
    if output.find(prompt) >= 0:
        output = output.replace(prompt, "")
    else:
        output = output[len(prompt) - 30 :]
        last_word = prompt[prompt.rfind(" ") + 1 :]
        output = output[output.find(last_word) + len(last_word) :]
    return output


def local_inference_api(prompt, max_tokens=3):
    API_URL = f"http://localhost:8000/inference"
    query_input = {"prompt": prompt, "max_length": max_tokens}

    response = requests.post(API_URL, json=query_input)
    output = response.json()

    output = output["output"]
    if output.find(prompt) >= 0:
        output = output.replace(prompt, "")
    else:
        last_word = prompt[prompt.rfind(" ") + 1 :]
        if output.find(last_word) >= 0:
            output = output[len(prompt) - 30 :]
            output = output[output.find(last_word) + len(last_word) :]
    return output


def gooseai_inference_api(prompt, max_tokens=3):
    API_URL = f"https://api.goose.ai/v1/engines/gpt-neo-20b/completions"
    query_input = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "temperature": 0.1,
    }

    headers = {"Authorization": f"Bearer {os.environ.get('GOOSEAI_KEY')}"}
    try:
        response = requests.post(API_URL, headers=headers, json=query_input)
        output = response.json()
        output = output["choices"][0]["text"]

        return output
    except Exception:
        return ""


def cohere_inference_api(prompt, max_tokens=3):
    co = cohere.Client(os.environ.get("COHERE_API_KEY"))
    try:
        response = co.generate(
            model="xlarge",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            k=0,
            p=0.9,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=[],
            return_likelihoods="NONE",
        )
        time.sleep(0.5)
    except Exception:
        return ""

    return response.generations[0].text


def ai21_inference_api(prompt, max_tokens=3):
    response = requests.post(
        "https://api.ai21.com/studio/v1/j1-jumbo/complete",
        headers={"Authorization": f"Bearer {os.environ.get('AI21_API_KEY')}"},
        json={
            "prompt": prompt,
            "numResults": 1,
            "maxTokens": max_tokens,
            "temperature": 0.9,
            "topKReturn": 0,
            "topP": 0.9,
            "stopSequences": [],
        },
    )
    output = response.json()
    output = output["completions"][0]["data"]["text"]

    return output


def hf_inference_local(prompt, model_name, model, tokenizer, max_length=3):
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.cuda()
    if model_name not in ["t5-11b", "ul2"]:
        max = encoded["attention_mask"].shape[1] + max_length
    else:
        max = max_length
    outputs = model.generate(input_ids, max_length=max)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    gc.collect()
    output = output.replace(prompt, "")
    return output


def load_model(model_name, api=True):
    if model_name.startswith("text"):
        return partial(gpt3, model_name=model_name)
    if api:
        if (
            model_name.find("opt-") >= 0
            or model_name == "ul2"
            or model_name == "EleutherAI/gpt-j-6B"
            or model_name == "google/flan-t5-xxl"
        ):
            return partial(local_inference_api)
        if model_name == "gpt-neox-20b":
            return partial(gooseai_inference_api)
        if model_name == "co:here":
            return partial(cohere_inference_api)
        if model_name == "jurassic1-jumbo":
            return partial(ai21_inference_api)
        return partial(hf_inference_api, model_name=model_name)
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        if "pt" in model_name:
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForSeq2SeqLM.from_config(config)
    settings = MODEL_SETTINGS[model_name]
    if "pt" in model_name:
        if model_name.find("opt") >= 0:
            device_map = infer_auto_device_map(
                model.model,
                no_split_module_classes=settings["no_split_module_classes"],
                dtype="float16",
            )
            if model_name == "facebook/opt-30b":
                #    device_map["decoder.embed_tokens.weight"] = 0
                device_map["decoder.layers.14"] = 3
                device_map["decoder.layers.15"] = 3
                device_map["decoder.layers.33"] = 3
                device_map["decoder.layers.32"] = 3
                device_map["decoder.layers.46"] = 3
                device_map["decoder.layers.47"] = 3

            if model_name == "facebook/opt-13b":
                device_map["decoder.layers.29"] = 2
                device_map["decoder.layers.30"] = 2
                device_map["decoder.layers.31"] = 2
                device_map["decoder.layers.32"] = 2
                device_map["decoder.layers.33"] = 2
                device_map["decoder.layers.34"] = 2
                device_map["decoder.layers.34.self_attn"] = 2
                device_map["decoder.layers.34.activation_fn"] = 2
                device_map["decoder.layers.34.fc1"] = 2
                device_map["decoder.layers.34.fc2"] = 2
                device_map["decoder.layers.34.self_attn_layer_norm"] = 2
                device_map["decoder.layers.34.final_layer_norm"] = 2

            model.model = load_checkpoint_and_dispatch(
                model.model,
                settings["checkpoint"],
                device_map=device_map,
                no_split_module_classes=settings["no_split_module_classes"],
                offload_folder="models_offload",
                offload_state_dict=True,
            )

            model.lm_head = load_checkpoint_and_dispatch(
                model.lm_head, settings["checkpoint"]
            )

            model.tie_weights()
        else:
            device_map = {
                "transformer.wte": 0,
                "transformer.drop": 0,
                "transformer.h.0": 0,
                "transformer.h.1": 0,
                "transformer.h.2": 0,
                "transformer.h.3": 0,
                "transformer.h.4": 0,
                "transformer.h.5": 0,
                "transformer.h.6": 0,
                "transformer.h.7": 0,
                "transformer.h.8": 0,
                "transformer.h.9": 0,
                "transformer.h.10": 0,
                "transformer.h.11": 0,
                "transformer.h.12": 0,
                "transformer.h.13": 0,
                "transformer.h.14": 0,
                "transformer.h.15": 0,
                "transformer.h.16": 0,
                "transformer.h.17": 0,
                "transformer.h.18": 0,
                "transformer.h.19": 0,
                "transformer.h.20": 0,
                "transformer.h.21": 0,
                "transformer.h.22": 1,
                "transformer.h.23": 1,
                "transformer.h.24": 1,
                "transformer.h.25": 1,
                "transformer.h.26": 1,
                "transformer.h.27": 1,
                "transformer.ln_f": 1,
                "lm_head": 1,
            }
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                offload_folder="models_offload2",
                offload_state_dict=True,
            )
    elif model_name == "google/flan-t5-xxl":
        model = T5ForConditionalGeneration.from_pretrained(
            settings["checkpoint"],
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            max_memory={0: "18GiB", 1: "10GiB"},
            device_map="auto",
        )
        model.tie_weights()
    else:
        device_map = "auto"
        if model_name == "ul2":
            print("USING UL2")
            device_map = {
                "shared": 0,
                "lm_head": 0,
                "encoder": 0,
                "decoder.embed_tokens": 1,
                "decoder.block.0": 1,
                "decoder.block.1": 1,
                "decoder.block.2": 1,
                "decoder.block.3": 1,
                "decoder.block.4": 1,
                "decoder.block.5": 1,
                "decoder.block.6": 1,
                "decoder.block.7": 1,
                "decoder.block.8": 1,
                "decoder.block.9": 1,
                "decoder.block.10": 1,
                "decoder.block.11": 1,
                "decoder.block.12": 1,
                "decoder.block.13": 1,
                "decoder.block.14": 1,
                "decoder.block.15": 1,
                "decoder.block.16": 1,
                "decoder.block.17": 1,
                "decoder.block.18": 1,
                "decoder.block.19": 1,
                "decoder.block.20": 1,
                "decoder.block.21": 1,
                "decoder.block.22": 1,
                "decoder.block.23": 1,
                "decoder.block.24": 1,
                "decoder.block.25": 1,
                "decoder.block.26": 2,
                "decoder.block.27": 2,
                "decoder.block.28": 2,
                "decoder.block.29": 2,
                "decoder.block.30": 2,
                "decoder.block.31": 2,
                "decoder.final_layer_norm": 2,
                "decoder.dropout": 2,
            }
        model = T5ForConditionalGeneration.from_pretrained(
            settings["checkpoint"],
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        model.tie_weights()
    if model_name == "facebook/opt-13b":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return partial(
        hf_inference_local, model_name=model_name, model=model, tokenizer=tokenizer
    )
