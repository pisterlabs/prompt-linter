import functools
import logging
import multiprocessing
import os
import sys
import time
from typing import Optional, Sequence, Union

import fire
import helm.common.request
import tqdm
from helm.common.authentication import Authentication
from helm.common.request import Request, RequestResult
from helm.proxy.accounts import Account
from helm.proxy.models import MODEL_NAME_TO_MODEL
from helm.proxy.services.remote_service import RemoteService
from openai import openai_object

from ml_swissknife import utils

from . import openai_utils

StrOrCompletionObject = Union[str, openai_object.OpenAIObject, helm.common.request.Sequence]

crfm_model_names = tuple(MODEL_NAME_TO_MODEL.keys())


def convert_crfm_object_to_openai_object(
    data: helm.common.request.Sequence,
) -> openai_object.OpenAIObject:
    """Convert helm.common.request.Sequence object to openai_object.OpenAIObject object."""
    return_data = openai_utils.convert_dict_to_openai_object(
        dict(
            text=data.text,
            logprobs=openai_utils.convert_dict_to_openai_object(
                dict(
                    tokens=[token.text for token in data.tokens],
                    top_logprobs=[
                        openai_utils.convert_dict_to_openai_object(token.top_logprobs) for token in data.tokens
                    ],
                    token_logprobs=[token.logprob for token in data.tokens],
                )
            ),
        )
    )
    return return_data


def normalize_model_name(model_name):
    # All CRFM model names have the org prefix. Prepend org name if it's not already in the model_name.
    if "/" in model_name:
        return model_name
    suffixes = [model_name.split("/")[1] for model_name in crfm_model_names]
    index = suffixes.index(model_name)
    model_name = crfm_model_names[index]
    return model_name


def crfm_quota(crfm_api_key: Optional[str] = None):
    crfm_api_key = os.getenv("CRFM_API_KEY") if crfm_api_key is None else crfm_api_key
    auth = Authentication(api_key=crfm_api_key)
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
    return account.usages


def crfm_completion_helper(
    prompt,
    service,
    model_name,
    top_k_per_token,
    random,
    sleep_time,
    auth,
    decoding_args: openai_utils.OpenAIDecodingArgumentsBase,
    stop_sequences,
):
    while True:
        try:
            request = Request(
                model=model_name,
                prompt=prompt,
                echo_prompt=decoding_args.echo,
                temperature=decoding_args.temperature,
                num_completions=decoding_args.n,
                max_tokens=decoding_args.max_tokens,
                stop_sequences=stop_sequences,
                top_p=decoding_args.top_p,
                presence_penalty=decoding_args.presence_penalty,
                frequency_penalty=decoding_args.frequency_penalty,
                top_k_per_token=top_k_per_token,
                random=random,
            )
            request_result: RequestResult = service.make_request(auth, request)
            break
        except Exception as e:
            logging.warning(f"Original exception: {e}.")
            logging.warning(f"Retrying request after {sleep_time} seconds.")
            time.sleep(sleep_time)  # Annoying rate limit on requests.
    return request_result.completions


def crfm_completion(
    prompts: Union[str, Sequence[str]],
    decoding_args: openai_utils.OpenAIDecodingArgumentsBase,
    model_name="text-davinci-003",
    sleep_time=2,
    max_instances=sys.maxsize,
    return_text=False,
    return_openai_object=True,
    crfm_api_key: Optional[str] = None,
    random: Optional[str] = None,
    num_procs: int = 1,
    **unused_kwargs,
) -> Union[StrOrCompletionObject, Sequence[StrOrCompletionObject], Sequence[Sequence[StrOrCompletionObject]],]:
    """Mirrors `openai_utils._openai_completion`.

    Args:
        prompts: A string or a list of strings to complete.
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        max_instances: Maximum number of prompts to decode.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        return_openai_object: If True and not return_text, return objects in the OpenAI (opposed to CRFM) format.
        crfm_api_key: CRFM API key.
        random: Random seed.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - a helm.common.request.Sequence object (if return_text is False and return_openai_object is False)
            - an openai_object.OpenAIObject object (if return_text is False and return_openai_object is True)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    utils.handle_unused_kwargs(unused_kwargs, msg="crfm_completion")

    crfm_api_key = os.getenv("CRFM_API_KEY") if crfm_api_key is None else crfm_api_key
    auth = Authentication(api_key=crfm_api_key)
    service = RemoteService("https://crfm-models.stanford.edu")

    is_single_prompt = isinstance(prompts, str)
    if is_single_prompt:
        prompts = [prompts]

    prompts = prompts[:max_instances]
    model_name = normalize_model_name(model_name)
    stop_sequences = [] if decoding_args.stop is None else list(decoding_args.stop)
    top_k_per_token = 1 if decoding_args.logprobs is None else decoding_args.logprobs

    with multiprocessing.Pool(num_procs) as p:
        partial_crfm_completion_helper = functools.partial(
            crfm_completion_helper,
            service=service,
            model_name=model_name,
            top_k_per_token=top_k_per_token,
            random=random,
            sleep_time=sleep_time,
            auth=auth,
            decoding_args=decoding_args,
            stop_sequences=stop_sequences,
        )
        completions = list(
            tqdm.tqdm(
                p.imap(partial_crfm_completion_helper, prompts),
                desc="prompt_batches",
                total=len(prompts),
            )
        )
        # flatten completions
        completions = [item for sublist in completions for item in sublist]

    if return_openai_object:
        completions = [convert_crfm_object_to_openai_object(completion) for completion in completions]
    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def main(**kwargs):
    # python -m ml_swissknife.crfm_utils
    quota = crfm_quota()
    print(quota)
    breakpoint()
    out = crfm_completion(
        prompts=["Life is"],
        decoding_args=openai_utils.OpenAIDecodingArguments(n=1, logprobs=3),
        return_text=False,
        return_openai_object=True,
        random="2000",
    )
    print(out[0].logprobs.top_logprobs)
    print(out[0].logprobs.tokens)
    print(out[0].logprobs.token_logprobs)
    breakpoint()
    out = crfm_completion(
        prompts="Life is",
        decoding_args=openai_utils.OpenAIDecodingArguments(n=2),
        return_text=True,
    )
    print(out)
    out = crfm_completion(
        prompts="Life is",
        decoding_args=openai_utils.OpenAIDecodingArguments(n=1),
        return_text=True,
    )
    print(out)


if __name__ == "__main__":
    fire.Fire(main)
