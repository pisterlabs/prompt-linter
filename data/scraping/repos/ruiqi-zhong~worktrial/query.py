import os
from typing import List, Dict, Union
from tqdm import tqdm
import time
import openai
import os
import asyncio
from copy import deepcopy
from typing import List, Dict, Iterator
import concurrent.futures
import numpy as np
from transformers import GPT2Tokenizer
import random

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OPENAI_ORG"]
GPT2TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")


MAX_LIMIT = 5
MAX_RETRIES = 20
BATCH_SIZE = 20

DEFAULT_MESSAGE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": None},
]


def get_length_in_gpt2_tokens(text: str) -> int:
    """
    Get the length of a text in GPT2 tokens.

    Parameters
    ----------
    text : str
        The text.

    Returns
    -------
    int
        The length of the text in GPT2 tokens.
    """
    return len(GPT2TOKENIZER.encode(text))


def truncate_based_on_gpt2_tokens(text: str, max_length: int) -> str:
    tokens = GPT2TOKENIZER.encode(text)
    if len(tokens) <= max_length:
        return text
    else:
        return GPT2TOKENIZER.decode(tokens[:max_length])


def get_avg_length(texts: List[str], max_num_samples=500) -> float:
    """
    Get the average length of texts in a list of texts.

    Parameters
    ----------
    texts : List[str]
        A list of texts.
    max_num_samples : int
        The maximum number of texts to sample to compute the average length.

    Returns
    -------
    float
        The average length of texts.
    """
    if len(texts) > max_num_samples:
        sampled_texts = random.sample(texts, max_num_samples)
    else:
        sampled_texts = texts
    avg_length = np.mean([get_length_in_gpt2_tokens(t) for t in sampled_texts])
    return avg_length


# hyperparameters
# in expectation the prompt will have the length (CONTEXT_LENGTH - CORPUS_OVERHEAD) * (1 - CORPUS_BUFFER_FRACTION) to leave room for the overflow and the completion
CORPUS_OVERHEAD = 1024
CORPUS_BUFFER_FRACTION = 0.25


def get_max_num_samples_in_proposer(texts: List[str], proposer_model: str) -> int:
    """
    Get the maximal number of in-context samples based on the context length.Leave a buffer of 25% of the relative context length and 1024 tokens for the absolute context length

    Parameters
    ----------
    texts : List[str]
        A list of texts.

    proposer_model : str
        The model used to propose descriptions.

    Returns
    -------
    int
        The maximal number of in-context samples.
    """
    max_corpus_pair_length = (get_context_length(proposer_model) - CORPUS_OVERHEAD) * (
        1 - CORPUS_BUFFER_FRACTION
    )
    avg_length = get_avg_length(texts)
    max_num_samples = int(max_corpus_pair_length / avg_length)
    return max_num_samples


def estimate_querying_cost(
    num_prompt_toks: int, num_completion_toks: int, model: str
) -> float:
    """
    Estimate the cost of running the API, as of 2023-04-06.

    Parameters
    ----------
    num_prompt_toks : int
        The number of tokens in the prompt.
    num_completion_toks : int
        The number of tokens in the completion.
    model : str
        The model to be used.

    Returns
    -------
    float
        The estimated cost of running the API.
    """

    if model == "gpt-3.5-turbo":
        cost_per_prompt_token = 0.002 / 1000
        cost_per_completion_token = 0.002 / 1000
    elif model == "gpt-4":
        cost_per_prompt_token = 0.03 / 1000
        cost_per_completion_token = 0.06 / 1000
    elif model == "gpt-4-32k":
        cost_per_prompt_token = 0.06 / 1000
        cost_per_completion_token = 0.12 / 1000
    else:
        raise ValueError(f"Unknown model: {model}")

    cost = (
        num_prompt_toks * cost_per_prompt_token
        + num_completion_toks * cost_per_completion_token
    )
    return cost


def get_context_length(model: str) -> int:
    """
    Get the context length for the given model.

    Parameters
    ----------
    model : str
        The model in the API to be used.

    Returns
    -------
    int
        The context length.
    """

    if model == "gpt-4":
        return 8000
    elif model == "gpt-4-32k":
        return 32000
    elif model == "gpt-3.5-turbo":
        return 4096
    elif model == "claude-v1.3":
        return 8192
    else:
        raise ValueError(f"Unknown model {model}")


def construct_claude_prompt_multi_turn(turns):
    import anthropic

    prompt = ""

    for turn in turns:
        if turn["role"] == "system":
            prompt += f"{anthropic.AI_PROMPT} {turn['content']}"
        elif turn["role"] == "user":
            prompt += f"{anthropic.HUMAN_PROMPT} {turn['content']}"
        else:
            raise ValueError(f"Unknown role {turn['role']}")
    assert turns[-1]["role"] == "user"

    prompt += f"{anthropic.AI_PROMPT}"
    return prompt


async def query_claude_once(client, semaphore, **args) -> Dict[str, str]:
    import anthropic

    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with semaphore:
                actual_prompt = args["prompt"]
                if type(actual_prompt) == list:
                    actual_prompt = construct_claude_prompt_multi_turn(actual_prompt)
                elif anthropic.HUMAN_PROMPT not in actual_prompt:
                    actual_prompt = (
                        f"{anthropic.HUMAN_PROMPT} {actual_prompt}{anthropic.AI_PROMPT}"
                    )

                resp = await client.acompletion(
                    prompt=actual_prompt,
                    stop_sequences=[],  # [anthropic.HUMAN_PROMPT],
                    model=args["model"],
                    max_tokens_to_sample=args["max_tokens_to_sample"],
                    temperature=args["temperature"],
                    top_p=args["top_p"],
                )
                resp["TMP_ID"] = args["TMP_ID"]
                return resp
        except Exception as e:
            retries += 1
            print(f"Error: {e}. Retrying {retries}/{MAX_RETRIES}")

    return {"completion": "", "TMP_ID": args["TMP_ID"]}


DEFAULT_CLAUDE_PARAMETER_DICT = {
    "model": "claude-v1.3",
    "max_tokens_to_sample": 100,
    "temperature": 0.7,
    "top_p": 1.0,
}


async def run_w_arg_list(args_list, progress_bar=False, max_concurrent=5):
    import anthropic

    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(args_list), desc="querying claude") if progress_bar else None

    tasks = []

    for arg in args_list:
        client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
        task = asyncio.create_task(query_claude_once(client, semaphore, **arg))
        task.add_done_callback(lambda x: pbar.update(1) if pbar is not None else None)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x["TMP_ID"])
    return [result["completion"] for result in results]


def claude_query(**args) -> List[str]:
    args_list = []
    for TMP_ID, prompt in enumerate(args["prompts"]):
        arg = deepcopy(DEFAULT_CLAUDE_PARAMETER_DICT)
        arg.update(args)
        arg["prompt"] = prompt
        del arg["prompts"]
        arg["TMP_ID"] = TMP_ID
        args_list.append(arg)

    result = asyncio.run(
        run_w_arg_list(
            args_list,
            progress_bar=args.get("progress_bar", False),
            max_concurrent=args.get("max_concurrent", 5),
        )
    )
    return result


DEFAULT_MESSAGE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": None},
]


def single_chat_gpt_wrapper(args) -> Union[None, str]:
    if args.get("messages") is None:
        args["messages"] = deepcopy(DEFAULT_MESSAGE)
        args["messages"][1]["content"] = args["prompt"]
        del args["prompt"]

    for _ in range(10):
        try:
            response = openai.ChatCompletion.create(**args)
            text_content_response = response.choices[0].message.content
            return text_content_response
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(e)
            time.sleep(10)

    return None


def chat_gpt_wrapper_parallel(
    prompts: List[str], num_processes: int = 1, progress_bar: bool = True, **args
) -> List[str]:
    def update_progress_bar(future):
        if progress_bar:
            pbar.update(1)

    if num_processes == 1:
        results = []
        pbar = tqdm(total=len(prompts), desc="Processing") if progress_bar else None
        for prompt in prompts:
            result = single_chat_gpt_wrapper({**args, "prompt": prompt})
            if progress_bar:
                pbar.update(1)
            results.append(result)
        if progress_bar:
            pbar.close()
        return results

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [
            executor.submit(single_chat_gpt_wrapper, {**args, "prompt": prompt})
            for prompt in prompts
        ]
        pbar = tqdm(total=len(tasks), desc="Processing") if progress_bar else None
        for task in concurrent.futures.as_completed(tasks):
            if progress_bar:
                task.add_done_callback(update_progress_bar)
        results = [task.result() for task in tasks]
    if progress_bar:
        pbar.close()
    return results


def query_wrapper(
    prompts: List[str],
    model: str = "claude-v1.3",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_processes: int = 1,
    progress_bar: bool = False,
) -> List[str]:
    """
    Wrapper for querying all LLM APIs.

    Parameters
    ----------
    prompts : List[str]
        List of prompts to query the model with.
    model : str, optional
        Model to query, by default "claude-v1.3"
    max_tokens : int, optional
        Maximum number of tokens to generate, by default 128
    temperature : float, optional
        Temperature for sampling, by default 0.7
    top_p : float, optional
        Top p for sampling, by default 1.0
    num_processes : int, optional
        Number of processes to use, by default 1

    Returns
    -------
    List[str]
        List of generated texts.
    """
    assert type(prompts) == list and len(prompts) > 0 and type(prompts[0]) != dict
    args = {}
    args["temperature"] = temperature
    args["top_p"] = top_p
    args["prompts"] = prompts

    if model.startswith("claude"):
        args["max_tokens_to_sample"] = max_tokens
        args["model"] = model
        args["max_concurrent"] = num_processes
        args["progress_bar"] = progress_bar

        return claude_query(**args)
    elif model.startswith("gpt"):
        args["model"] = model
        args["max_tokens"] = max_tokens
        args["num_processes"] = num_processes
        args["progress_bar"] = progress_bar
        return chat_gpt_wrapper_parallel(**args)
    else:
        raise ValueError(f"Unknown model {model}")


if __name__ == "__main__":
    prompt = [
        {"role": "user", "content": "hello world"},
        {"role": "system", "content": "what can I do for you?"},
        {"role": "user", "content": "I want to book a flight to Paris"},
    ]

    result = query_wrapper([prompt], model="claude-2.1", progress_bar=True)
    print(result)

    result = query_wrapper(
        ["hello world", "what can you do?"] * 20, model="claude-2.1", progress_bar=True
    )
    print(result)
