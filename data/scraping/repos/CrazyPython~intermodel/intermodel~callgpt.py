#!/usr/bin/env python
import re
import traceback
import asyncio
import cmd
import datetime
import os
import uuid
import hashlib
from typing import Union, List, Optional

import banana_dev as banana
import openai
import anthropic
import httpx
import tiktoken

import intermodel.callgpt_faker

from dotenv import load_dotenv

load_dotenv()

MODEL_ALIASES = {}
MODEL_TO_AUTHOR = {
    "flan-t5-xxl": "google",
}
MODEL_TO_BANANA_MODEL_KEY = {}


async def complete(
    model,
    prompt=None,
    temperature=None,
    top_p=None,
    max_tokens=None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Union[float, int] = 0,
    presence_penalty: Union[float, int] = 0,
    num_completions: int = None,
    top_k=None,
    repetition_penalty: Union[float, int] = 1,
    tfs=1,
    user_id=None,
    logit_bias=None,
    vendor=None,
    vendor_config=None,
    **kwargs,
):
    model = MODEL_ALIASES.get(model, model)
    # todo: multiple completions, top k, logit bias for all vendors
    # todo: detect model not found on all vendors and throw the same exception
    if vendor is None:
        vendor = pick_vendor(model, vendor_config)
    if vendor_config is not None and vendor in vendor_config:
        kwargs = {**vendor_config[vendor]["config"], **kwargs}
    if vendor.startswith("openai"):
        if user_id is None:
            hashed_user_id = None
        else:
            hash_object = hashlib.sha256()
            hash_object.update(os.getenv("INTERMODEL_HASH_SALT", "").encode("utf-8"))
            hash_object.update(str(user_id).encode("utf-8"))
            hashed_user_id = hash_object.hexdigest()

        rest = dict(kwargs)
        rest.pop("openai_api_key")
        api_arguments = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop if stop != [] else None,
            "user": hashed_user_id,
            "api_key": kwargs["openai_api_key"],
            "logit_bias": logit_bias,
            "n": num_completions,
            **rest,
        }
        # remove None values, OpenAI API doesn't like them
        for key, value in dict(api_arguments).items():
            if value is None:
                del api_arguments[key]
        if (
            model.startswith("gpt-3.5")
            or model.startswith("gpt-4")
            and not model.endswith("-base")
        ):
            api_arguments["messages"] = [
                {"role": "user", "content": api_arguments["prompt"]}
            ]
            if "prompt" in api_arguments:
                del api_arguments["prompt"]
            if "logprobs" in api_arguments:
                del api_arguments["logprobs"]
            api_response = await openai.ChatCompletion.acreate(**api_arguments)
            for c in api_response["choices"]:
                c["text"] = c["message"]["content"]
        else:
            api_response = await openai.Completion.acreate(**api_arguments)
        return {
            "prompt": {"text": prompt if prompt is not None else "<|endoftext|>"},
            "completions": [
                {
                    "text": completion.text,
                    "finish_reason": {
                        "reason": completion.get("finish_reason", "unknown")
                    },
                }
                for completion in api_response.choices
            ],
            "model": api_response.model,
            "id": api_response.id,
            "created": api_response.created,
            "usage": {
                # "prompt_tokens": api_response.usage.prompt_tokens,
                # # if the completion is empty, the value will be missing
                # "completion_tokens": api_response.usage.get("completion_tokens", 0),
                # "charged_tokens": api_response.usage.total_tokens,
                "vendor": vendor,
            },
        }
    elif vendor == "ai21":
        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://api.ai21.com/studio/v1/{model}/complete",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get("ai21_api_key", os.environ.get("AI21_API_KEY"))
                },
                json={
                    "prompt": prompt,
                    "numResults": num_completions or 1,
                    "maxTokens": max_tokens,
                    # "stopSequences": stop,
                    "topP": top_p,
                    "temperature": temperature,
                    "frequencyPenalty": {"scale": frequency_penalty},
                    "presencePenalty": {"scale": presence_penalty},
                    **kwargs,
                },
            )
        http_response.raise_for_status()
        api_response = http_response.json()
        completion_tokens = sum(
            [
                len(completion["data"]["tokens"])
                for completion in api_response["completions"]
            ]
        )
        return {
            "prompt": {"text": api_response["prompt"]["text"]},
            "completions": [
                {
                    "text": completion["data"]["text"],
                    "finish_reason": completion["finishReason"]["reason"],
                }
                for completion in api_response["completions"]
            ],
            "model": model,
            "id": api_response["id"],
            "usage": {
                "prompt_tokens": len(api_response["prompt"]["tokens"]),
                "completion_tokens": completion_tokens,
                "charged_tokens": completion_tokens,
                "vendor": vendor,
            },
        }
    elif vendor == "textsynth":
        pass
    elif vendor == "huggingface":
        # todo: freq and pres penalties
        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get(
                        "huggingface_api_key", os.environ.get("HUGGINGFACE_API_KEY")
                    )
                },
                json={
                    "inputs": prompt,
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    **kwargs,
                },
            )
    elif vendor == "banana":
        # this is not a complete implementation
        api_response = banana.run(
            kwargs.get("banana_api_key", os.getenv("BANANA_API_KEY")),
            MODEL_TO_BANANA_MODEL_KEY[model],
            model_inputs={
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                **kwargs,
            },
        )
        return {
            "prompt": {"text": prompt},
            "completions": [
                {"text": model_output["text"]}
                for model_output in api_response["modelOutputs"]
            ],
            "model": model,
            "id": api_response["id"],
            "created": api_response["created"],
            "usage": {
                # "completion_tokens": sum([len(output['generated_text_tokens']) for output in api_response['modelOutputs']]),
                "vendor": vendor,
            },
        }
    elif vendor == "forefront":
        if "t5-20b" in model:
            prompt = prompt + " <extra_id_0>"
        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://shared-api.{model}",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get("forefront_api_key", os.getenv("FOREFRONT_API_KEY"))
                },
                json={
                    "text": prompt,
                    "top_p": top_p,
                    "top_k": 50400 or top_k,
                    "temperature": temperature,
                    "tfs": tfs,
                    "length": max_tokens,
                    "repetition_penalty": repetition_penalty,
                    "stop": stop,
                },
            )
        http_response.raise_for_status()
        api_response = http_response.json()
        return {
            "prompt": {"text": prompt},
            "model": api_response["model"],
            "completions": [
                {"text": output["completion"]} for output in api_response["result"]
            ],
            "created": api_response["timestamp"],
            # forefront bills both the prompt and completion
            "usage": NotImplemented,
        }
    elif vendor == "anthropic":
        if num_completions not in [None, 1]:
            raise NotImplementedError("Anthropic only supports num_completions=1")
        client = anthropic.Client(
            kwargs.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
        )
        response = await client.acompletion(
            model=model,
            prompt=prompt or "\n\nHuman:",
            max_tokens_to_sample=max_tokens or 16,
            temperature=temperature or 1,
            top_p=top_p or 1,
            top_k=top_k or -1,
            stop_sequences=stop or list(),
            disable_checks=True,
            **kwargs,
        )
        if response["stop_reason"] == "stop_sequence":
            finish_reason = "stop"
        elif response["stop_reason"] == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "unknown"
        return {
            "prompt": {
                "text": prompt,
            },
            "completions": [
                {"text": response["completion"], "finish_reason": finish_reason}
            ],
            "model": model,
            "id": uuid.uuid4(),
            "created": datetime.datetime.now(),
            "usage": {
                "charged_tokens": 0,
                "vendor": vendor,
            },
        }
    elif vendor == "replicate":
        async with httpx.AsyncClient() as client:
            initial_response = await client.post(
                "https://api.replicate.com/v1/predictions",
                json={
                    "version": model.split(":")[1],
                    "prompt": prompt,
                    "max_length": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "stop_sequence": stop[0],
                },
            )
            initial_response.raise_for_status()
            initial_response_json = initial_response.json()
            response = initial_response
            while response.json()["status"] != "succeeded":
                await asyncio.sleep(0.25)
                response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{initial_response_json['id']}"
                )
                response.raise_for_status()
        return {
            "prompt": {"text": prompt},
            "completions": {
    
    }
        }
    elif vendor == "fake-local":
        return intermodel.callgpt_faker.fake_local(
            model=model,
            vendor=vendor,
            prompt=prompt,
            max_tokens=max_tokens,
            num_completions=num_completions,
        )
    else:
        raise NotImplementedError(f"Unknown vendor {vendor}")


def complete_sync(*args, **kwargs):
    return asyncio.run(complete(*args, **kwargs))


def tokenize(model: str, string: str) -> List[int]:
    model = MODEL_ALIASES.get(model, model)
    try:
        vendor = pick_vendor(model)
    except NotImplementedError:
        vendor = None
    if vendor == "openai" or model == "gpt2":
        # tiktoken internally caches loaded tokenizers
        tokenizer = tiktoken.encoding_for_model(model)
        # encode special tokens as normal
        # XXX: make this an option
        return tokenizer.encode(string, disallowed_special={})
    elif vendor == "anthropic":
        # anthropic caches the tokenizer
        # XXX: this may send synchronous network requests, could be downloaded as part of build
        tokenizer = anthropic.get_tokenizer()
        encoded_text = tokenizer.encode(string)
        return encoded_text.ids
    else:
        raise NotImplementedError(f"I don't know how to tokenize {model}")


def count_tokens(model: str, string: str) -> int:
    return len(tokenize(model, string))


def untokenize(model: str, string: List[int]) -> str:
    model = MODEL_ALIASES.get(model, model)
    try:
        vendor = pick_vendor(model)
    except NotImplementedError:
        vendor = None
    if vendor == "openai" or model == "gpt2":
        # tiktoken internally caches loaded tokenizers
        tokenizer = tiktoken.encoding_for_model(model)
        return tokenizer.decode(string)
    elif vendor == "anthropic":
        # anthropic caches the tokenizer
        # XXX: this may send synchronous network requests, could be downloaded as part of build
        tokenizer = anthropic.get_tokenizer()
        encoded_text = tokenizer.decode(string)
        return encoded_text.ids
    else:
        raise NotImplementedError(f"I don't know how to tokenize {model}")


def pick_vendor(model, custom_config=None):
    if custom_config is not None:
        for vendor_name, vendor in custom_config.items():
            if vendor["provides"] is not None:
                for pattern in vendor["provides"]:
                    if re.match(pattern, model):
                        return vendor_name
    model = MODEL_ALIASES.get(model, model)
    if (
        "ada" in model
        or "babbage" in model
        or "curie" in model
        or "davinci" in model
        or "cushman" in model
        or "text-moderation-" in model
        or model.startswith("ft-")
        or model.startswith("gpt-4")
        or model.startswith("gpt-3.5-")
    ):
        return "openai"
    elif "j1-" in model or model.startswith("j2-"):
        return "ai21"
    elif "forefront" in model:
        return "forefront"
    elif "t5" in model:
        # this should determine based on the specific model
        return "banana"
    elif model.startswith("claude-"):
        return "anthropic"
    else:
        raise NotImplementedError("Unknown model")


def max_token_length(model):
    """
    The maximum number of tokens in the prompt and completion
    """
    if model == "gpt-4-32k":
        return 32769
    elif model.startswith("gpt-4"):
        return 8193
    elif model == "code-davinci-002":
        return 8001
    elif model.startswith("code"):
        raise ValueError("Unknown maximum")
    elif model == "gpt-3.5-turbo-16k":
        return 16385
    elif model in ('babbage-002', 'davinci-002'):
        return 16385
    elif model == "gpt-3.5-turbo":
        return 4097
    elif model == "gpt-3.5-turbo-instruct":
        return 4097
    elif model == "text-davinci-003" or model == "text-davinci-002":
        return 4097
    elif model == "text-embedding-ada-002":
        return 8191
    elif model.startswith("text-embedding-") and model.endswith("-001"):
        return 2046
    elif (
        "ada" in model
        or "babbage" in model
        or "curie" in model
        or "davinci" in model
        or "cushman" in model
    ):
        return 2049
    else:
        raise NotImplementedError(f"Token cap not known for model {model}")


class InteractiveIntermodel(cmd.Cmd):
    prompt = "intermodel> "

    def do_c(self, arg):
        """
        Send a completion request to a model
        Usage: c <model> <prompt>
        """
        model = arg.split()[0]
        prompt = arg[arg.index(model) + len(model) + 1 :]
        try:
            print(complete_sync(model, pick_vendor(model), prompt))
        except NotImplementedError:
            print(f"Not implemented for model {model}")
            print(traceback.format_exc())

    def do_t(self, arg):
        """
        Tokenize a model
        Usage: t <model> <prompt>
        """
        model = arg.split()[0]
        prompt = arg[arg.index(model) + len(model) + 1 :]
        try:
            print(tokenize(model, prompt))
        except NotImplementedError:
            print(f"Not implemented for model {model}")
            print(traceback.format_exc())

    def do_EOF(self, arg):
        """Exit"""
        return True


if __name__ == "__main__":
    InteractiveIntermodel().cmdloop()
