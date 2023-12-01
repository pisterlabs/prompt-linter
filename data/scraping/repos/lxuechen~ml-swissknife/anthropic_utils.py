import copy
import functools
import logging
import math
import multiprocessing
import os
import sys
import time
from typing import Sequence, Union

import anthropic
import tqdm

from .openai_utils import OpenAIDecodingArguments, convert_dict_to_openai_object


def convert_args_oai_to_anthropic(decoding_args: dict):
    args_mapping = {
        "max_tokens": "max_tokens_to_sample",
        "stop": "stop_sequences",
    }
    for arg in args_mapping:
        if arg in decoding_args:
            decoding_args[args_mapping[arg]] = decoding_args[arg]
            del decoding_args[arg]
    if "stop_sequences" not in decoding_args or decoding_args["stop_sequences"] is None:
        decoding_args["stop_sequences"] = [anthropic.HUMAN_PROMPT]
    return decoding_args


def _anthropic_completion_helper(
    prompt_batch: Sequence[str],
    shared_kwargs: dict,
    sleep_time: int = 2,
) -> Sequence[dict[str, Union[str, float]]]:
    shared_kwargs = copy.deepcopy(shared_kwargs)
    client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    shared_kwargs = convert_args_oai_to_anthropic(shared_kwargs)

    while True:
        try:
            response = client.completion(prompt=prompt_batch[0], **shared_kwargs)

            if response["completion"] == "":
                response["completion"] = " "  # annoying doesn't allow empty string

            break
        except anthropic.api.ApiException as e:
            logging.warning(f"ApiException: {e}.")
            if "status code: 429" in str(e):
                logging.warning(f"Rate limit hit. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
            elif "exceeds max" in str(e):
                shared_kwargs["max_tokens_to_sample"] = int(shared_kwargs["max_tokens_to_sample"] * 0.8)
                logging.warning(f"Reducing target length to {shared_kwargs['max_tokens_to_sample']}, Retrying...")
            else:
                raise ValueError("Unknown ApiException.")
    # convert response to an openai object
    response["text"] = response["completion"]
    del response["completion"]
    response["logprobs"] = None
    response = convert_dict_to_openai_object(response)
    return [response]


def anthropic_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="claude-v1.3",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    num_procs=1,
    **decoding_kwargs,
) -> Union[Sequence[str], Sequence[dict[str, str]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    assert num_procs < 8, "Anthropic API only allows 8 concurrent requests."
    logging.info(f"Decoding with Anthropic API model {model_name} and numproc == {num_procs}.")
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    if batch_size > 1:
        logging.warning("Batching is not supported for Anthropic API. Setting batch_size to 1.")
        batch_size = 1

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    shared_kwargs = dict(
        model=model_name,
        **decoding_args.__dict__,
        **decoding_kwargs,
    )
    with multiprocessing.Pool(num_procs) as p:
        partial_completion_helper = functools.partial(
            _anthropic_completion_helper,
            shared_kwargs=shared_kwargs,
            sleep_time=sleep_time,
        )
        completions = list(
            tqdm.tqdm(
                p.imap(partial_completion_helper, prompt_batches),
                desc="prompt_batches",
                total=len(prompt_batches),
            )
        )
    # flatten the list
    completions = [completion for completion_batch in completions for completion in completion_batch]

    if return_text:
        completions = [completion["completion"] for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions
