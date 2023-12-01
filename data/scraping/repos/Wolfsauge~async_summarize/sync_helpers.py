import os
import sys
import re
import json
import math
from typing import Iterable, Tuple, TypeVar
import yaml

# from tqdm import tqdm  # type: ignore
# import rich.progress
import httpx
from transformers import AutoTokenizer, LlamaTokenizerFast  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from openai import AsyncOpenAI
import jinja2
from icecream import ic  # type: ignore

T = TypeVar("T")


def get_length_of_chunk_in_tokens(my_chunk: str, buck_slip: dict) -> int:
    my_result = buck_slip["tokenizer"](my_chunk)
    input_ids = my_result.input_ids
    length_of_chunk_in_tokens = len(input_ids)

    return length_of_chunk_in_tokens


def get_buck_slip_config(buck_slip_filename: str) -> dict:
    buck_slip = {
        "httpx_max_connections": 1,
        "httpx_max_keepalive_connections": 1,
        "model_identifier": "empty",
        "api_key": "empty",
    }

    try:
        ic(buck_slip_filename)
        with open(buck_slip_filename, "r", encoding="utf-8") as file:
            buck_slip = yaml.safe_load(file)
        ic(buck_slip)

    except (IOError, OSError) as my_exception:
        warning_message = f"{my_exception}"
        ic(warning_message)
    buck_slip["model_local_identifier"] = str(buck_slip["model_identifier"]).replace(
        "/", "_"
    )

    return buck_slip


def get_prompt_template(prompt_template_filename: str) -> str:
    try:
        ic(prompt_template_filename)
        with open(prompt_template_filename, "r", encoding="utf-8") as file:
            prompt_template = yaml.safe_load(file)
            prompt_template = prompt_template["prompt_templates"]
        # Create enum of tasks (summarize, merge)
        # Validate for each task a prompt is available
        # Otherwise error out
        ic(prompt_template)

    except (IOError, OSError) as my_exception:
        warning_message = f"{my_exception}"
        ic(warning_message)

    return prompt_template


def get_tokenizer(buck_slip: dict) -> LlamaTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        buck_slip["model_identifier"], use_fast=buck_slip["use_fast"]
    )
    ic(type(tokenizer))
    ic(tokenizer.is_fast)
    buck_slip["tokenizer.is_fast"] = tokenizer.is_fast

    encoding = tokenizer("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    ic(type(encoding))
    ic(encoding.is_fast)
    buck_slip["encoding.is_fast"] = encoding.is_fast

    return tokenizer


def get_text_splitter(
    buck_slip: dict, custom_chunk_size: int, custom_chunk_overlap: int
) -> TextSplitter:
    batched_tokenization = buck_slip["use_batched_tokenization"]

    if batched_tokenization is True:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=buck_slip["tokenizer"],
            chunk_size=custom_chunk_size,
            chunk_overlap=custom_chunk_overlap,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=custom_chunk_size,
            chunk_overlap=custom_chunk_overlap,
            length_function=lambda x: get_length_of_chunk_in_tokens(x, buck_slip),
        )

    return text_splitter


def get_api_client(buck_slip: dict) -> AsyncOpenAI:
    my_max_keepalive_connections = int(buck_slip["httpx_max_keepalive_connections"])
    my_max_connections = int(buck_slip["httpx_max_connections"])
    limits = httpx.Limits(
        max_keepalive_connections=my_max_keepalive_connections,
        max_connections=my_max_connections,
    )
    timeout = httpx.Timeout(1200.0, connect=60.0)

    api_client = AsyncOpenAI(
        api_key=buck_slip["api_key"],
        base_url=buck_slip["api_url"],
        http_client=httpx.AsyncClient(limits=limits, timeout=timeout),
    )
    ic(type(api_client))

    return api_client


def get_jinja2_environment():
    jinja2_env = jinja2.Environment()
    ic(type(jinja2_env))

    return jinja2_env


def get_file_contents(my_filename: str, buck_slip: dict) -> str:
    with open(my_filename, "r", encoding="utf-8") as my_fp:
        sample_text = my_fp.read()
    buck_slip["length_of_sample_text_in_characters"] = len(sample_text)
    ic(len(sample_text))

    return sample_text


def get_output_filename(my_input_filename: str, buck_slip: dict) -> str:
    my_index = 0
    does_exist = False
    while my_index < 1000:
        my_index_str = f"{my_index:04d}"
        my_local_identifier = buck_slip["model_local_identifier"]
        replacement = f"-analysis-{my_local_identifier}-{my_index_str}.json"
        output_filename = os.path.basename(my_input_filename)
        output_filename = re.sub("\\.txt$", replacement, output_filename)
        does_exist = os.path.exists(output_filename)
        if does_exist is True:
            my_index += 1
        else:
            break

    if does_exist is True:
        ic("ERROR: Can't find output filename.")
        sys.exit(1)

    ic(output_filename)

    return output_filename


def insert_buckslip_into_result(result: dict, buck_slip: dict) -> dict:
    # Stringify non JSON serializable elements of the buck slip
    buck_slip["tokenizer_str"] = str(buck_slip["tokenizer"])
    buck_slip["text_splitter_str"] = str(buck_slip["text_splitter"])
    buck_slip["api_client_str"] = str(buck_slip["api_client"])
    buck_slip["jinja2_env_str"] = str(buck_slip["jinja2_env"])
    buck_slip["lock_str"] = str(buck_slip["lock"])
    del buck_slip["tokenizer"]
    del buck_slip["text_splitter"]
    del buck_slip["api_client"]
    del buck_slip["jinja2_env"]
    del buck_slip["lock"]

    # Insert stringified and thus JSON serializable buck slip into the result dict
    result["buck_slip"] = buck_slip

    return result


def write_output_file(output_filename: str, data: dict) -> None:
    with open(output_filename, "w", encoding="utf-8") as my_fp:
        json.dump(data, my_fp)
    ic(output_filename)


def grouped(iterable: Iterable[T], number_of_elements=2) -> Iterable[Tuple[T, ...]]:
    """https://stackoverflow.com/a/5389547"""
    return zip(*[iter(iterable)] * number_of_elements)


def power_log(my_x: int) -> int:
    """https://stackoverflow.com/a/14267825"""
    return 2 ** (math.ceil(math.log(my_x, 2)))


def find_chunk_pair_with_minimal_size(elements) -> tuple[int, int]:
    last_index = len(elements) - 1
    min_length = len(elements[0])
    min_index = 0
    for i, result in enumerate(elements):
        if i < last_index:
            sum_of_chars = len(result) + len(elements[i + 1])
            if sum_of_chars < min_length:
                min_length = sum_of_chars
                min_index = i
    return min_index, min_index + 1


def find_longest_element_index(elements) -> int:
    max_length = 0
    max_index = 0
    for i, result in enumerate(elements):
        if len(result) > max_length:
            max_length = len(result)
            max_index = i
    return max_index


def calc_custom_chunking_parameters(
    length_of_chunk_in_tokens: int, buck_slip: dict
) -> tuple[int, int]:
    my_divisor = math.ceil(length_of_chunk_in_tokens / buck_slip["chunk_size"])
    my_divisor = power_log(my_divisor)
    my_custom_chunk_size = math.ceil(length_of_chunk_in_tokens / my_divisor)
    my_custom_chunk_size = math.ceil(my_custom_chunk_size * 1.10)

    my_custom_chunk_overlap = math.ceil(my_custom_chunk_size * 0.1)

    return my_custom_chunk_size, my_custom_chunk_overlap
