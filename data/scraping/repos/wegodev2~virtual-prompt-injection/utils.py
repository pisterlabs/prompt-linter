import os
import io
import time
import json
import asyncio
from typing import Optional, Sequence, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from tqdm import tqdm
import openai


openai.api_key = "sk-"  # PUT YOUR OPENAI API KEY HERE


@dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[dict] = field(default_factory=dict)


async def dispatch_openai_requests(
    messages_list: List[List[Dict[str, Any]]],
    decoding_args: OpenAIDecodingArguments,
    model_name: str,
) -> List[str]:
    shared_kwargs = dict(
        model=model_name,
        **decoding_args.__dict__
    )
    async_responses = [
        openai.ChatCompletion.acreate(
            messages=x,
            **shared_kwargs
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def openai_complete(
    prompt_lst: List,
    decoding_args: OpenAIDecodingArguments,
    model_name: str,
    batch_size: int = 10
) -> Tuple[List[str], List[str], int, float]:
    request_start = time.time()
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    message_list = []
    for prompt in prompt_lst:
        if (model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4")):
            message = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        else:
            message = prompt
        message_list.append(message)
    prediction_lst = []
    finish_reason_lst = []
    i = 0
    wait_base = 10
    retry = 0
    progress_bar = tqdm(total=len(message_list))
    original_max_tokens = decoding_args.max_tokens
    while (i < len(message_list)):
        try:
            if (model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4")):
                batch_responses = asyncio.run(
                    dispatch_openai_requests(
                        messages_list=message_list[i:i + batch_size],
                        decoding_args=decoding_args,
                        model_name=model_name
                    )
                )
                for response in batch_responses:
                    prediction_lst.append(response['choices'][0]['message']['content'])
                    finish_reason_lst.append(response['choices'][0]['finish_reason'])
                    total_tokens += response['usage']['total_tokens']
                    total_prompt_tokens += response['usage']['prompt_tokens']
                    total_completion_tokens += response['usage']['completion_tokens']
                progress_bar.update(len(batch_responses))
            elif model_name == 'text-davinci-003':
                response = openai.Completion.create(
                    model=model_name,
                    prompt=message_list[i:i + batch_size],
                    **decoding_args.__dict__
                )
                batch_predictions = [""] * len(response['choices'])
                batch_finish_reasons = [""] * len(response['choices'])
                for choice in response['choices']:
                    batch_predictions[choice['index']] = choice['text']
                    batch_finish_reasons[choice['index']] = choice['finish_reason']
                prediction_lst += batch_predictions
                finish_reason_lst += batch_finish_reasons
                total_tokens += response['usage']['total_tokens']
                total_prompt_tokens += response['usage']['prompt_tokens']
                total_completion_tokens += response['usage']['completion_tokens']
                progress_bar.update(len(batch_predictions))
            i += batch_size
            # reset hyperparameters
            wait_base = 10
            retry = 0
            decoding_args.max_tokens = original_max_tokens
        except openai.error.OpenAIError as e:
            print(repr(e))
            retry += 1
            print("Batch error: ", i, i + batch_size)
            print("retry number: ", retry)
            if "Please reduce" in str(e):
                decoding_args.max_tokens = int(decoding_args.max_tokens * 0.8)
                print(f"Reducing target length to {decoding_args.max_tokens}, Retrying...")
            else:
                print(f"Hit request rate limit; retrying...; sleep ({wait_base})")
                time.sleep(wait_base)
                wait_base = wait_base * 2
    request_duration = time.time() - request_start
    print(f"Generated {len(message_list)} responses in {request_duration:.2f}s")
    if model_name.startswith("gpt-3.5-turbo"):
        cost = 0.0015 * total_prompt_tokens + 0.002 * total_completion_tokens
    elif model_name.startswith("gpt-4"):
        cost = 0.03 * total_prompt_tokens + 0.06 * total_completion_tokens
    elif model_name == 'text-davinci-003':
        cost = 0.02 * total_tokens
    else:
        cost = 0
    return prediction_lst, finish_reason_lst, total_tokens, cost / 1000


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
