# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import math
import re
import sys
import traceback
from collections.abc import Iterable, Set
from typing import Annotated, cast

import aiohttp
import more_itertools
import numpy as np
import openai
import scipy.special
import tqdm
import typer
from datasets import DatasetDict, load_dataset
from lmapi.lm import LM, CompletionsSettings
from lmapi.openai import client_session

from dp_few_shot_generation.lm import (
    api_openai_com,
    next_logprobs,
    normalized_logprobs_for_chosen_tokens,
)
from dp_few_shot_generation.prob_utils import densify, log_max_normalize, log_normalize

DEFAULT_NUM_PRIVATE_TRAIN = 80
DEFAULT_NUM_PUBLIC_TRAIN = 0
DEFAULT_NUM_VALID = 4
DEFAULT_NUM_PRIVATE_TRAIN_SPLITS = 20
DEFAULT_NUM_TEST = -1


def format_full_datum_for_prompt(field_name, datum: dict[str, str]):
    return (
        f'{field_name}: "{datum["label"]}"\nSentence: "{datum["content"] + " END"}"\n'
    )


def format_test_input_for_prompt(field_name, test_input: str):
    return f'{field_name}: "{test_input}"\nSentence: "'


def construct_prompt_same(train_examples, test_example, field_name):
    prompt = f""  # prompt strucrture follows: https://github.com/tonyzhaozh/few-shot-learning/blob/main/data_utils.py#L427-L429
    for train_example in train_examples:
        prompt += "Sentence: " + train_example["content"] + "\n"
        prompt += f"{field_name}: " + train_example["label"] + "\n\n"
    prompt += "Sentence: " + test_example["content"] + "\n"
    prompt += f"{field_name}:"
    return prompt


def complete(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                max_tokens=l,
                temperature=temp,
                logprobs=num_log_probs,
                echo=echo,
                stop="\n",
                n=n,
            )
            received = True
        except:
            error = sys.exc_info()[0]
            if (
                error == openai.error.InvalidRequestError
            ):  # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(1)
    return response


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_model_response(
    data,
    test_examples,
    openai_model,
    field_name,
    max_token_to_fill=5,
    additional_tokens=None,
):
    all_raw_answers = []

    prompts = []
    train_examples = data

    for test_example in test_examples:
        prompts.append(construct_prompt_same(train_examples, test_example, field_name))

    if additional_tokens is not None:
        assert len(additional_tokens) == len(prompts)
        for i in range(len(prompts)):
            prompts[i] += additional_tokens[i]

    chunked_prompts = list(chunks(prompts, 20))
    for test_chunk in chunked_prompts:
        response = complete(
            test_chunk, l=max_token_to_fill, model_name=openai_model, num_log_probs=100
        )

        for answer_id, answer in enumerate(response["choices"]):
            all_raw_answers.append(answer)

    return all_raw_answers


def em_accuracy_helper(prediction, label):
    correctness_list = []
    for pred, l in zip(prediction, label):
        pred = pred.split("\n")[0]
        if pred == l:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)


def merge_logprobs_topk_mean(
    private_next_logprobs: list[dict[int, float]],
    public_next_logprobs: dict[int, float],
    n_vocab: int,
    no_public_token: bool,
    normalize_max: bool,
) -> np.ndarray:
    # Compute merged distribution
    # logsumexp - np.log(...): compute mean probability of distribution
    if normalize_max:
        normalize_func = (
            log_max_normalize  # normalize max probability to 1, Exponential mechanism
        )
    else:
        normalize_func = (
            log_normalize  # normalize sum probability to 1, Gaussian mechanism
        )
    if no_public_token:
        merged_next_logprobs = scipy.special.logsumexp(
            np.stack(
                [
                    # Turn into a 1D tensor of size n_vocab
                    densify(
                        n_vocab,
                        # Normalize distribution
                        normalize_func(
                            # Filter to the top 100 most likely next tokens according to the public prompt
                            {k: v for k, v in lps.items()}
                        ),
                    )
                    for lps in private_next_logprobs
                ]
            ),
            axis=0,
        ) - np.log(len(private_next_logprobs))

    else:
        merged_next_logprobs = scipy.special.logsumexp(
            np.stack(
                [
                    # Turn into a 1D tensor of size n_vocab
                    densify(
                        n_vocab,
                        # Normalize distribution
                        normalize_func(
                            # Filter to the top 100 most likely next tokens according to the public prompt
                            {k: v for k, v in lps.items() if k in public_next_logprobs}
                        ),
                    )
                    for lps in private_next_logprobs
                ]
            ),
            axis=0,
        ) - np.log(len(private_next_logprobs))
    merged_next_probs = np.exp(merged_next_logprobs)
    return merged_next_probs


async def generate_with_private_prompts(
    trainset,
    num_private_train,
    num_private_train_splits,
    instruction,
    public_train_prompt: str,
    stop_tokens: Set[int],
    test_input: str,
    lm: LM,
    noise_rng: np.random.RandomState,
    sigma: float,
    field_name: str,
    top_p,
    no_public_token: bool,
    subsample_per_token: bool,
    gen_seed: int,
    max_tokens: int,
    normalize_max: bool = False,
) -> list[int]:
    generated_token_ids: list[int] = []

    stringified_test_datum = format_test_input_for_prompt(field_name, test_input)
    public_prompt = public_train_prompt + stringified_test_datum
    public_prompt_tokens = lm.encoding.encode(public_prompt)

    assert num_private_train_splits > 0

    train_subset = trainset.select(range(len(trainset)), keep_in_memory=True)

    if not subsample_per_token:
        private_train_subset = cast(
            Iterable[dict[str, str]],
            train_subset.shuffle(gen_seed, keep_in_memory=True).select(
                range(num_private_train), keep_in_memory=True
            ),
        )
        private_train_splits = [
            list(it)
            for it in more_itertools.distribute(
                num_private_train_splits, private_train_subset
            )
        ]
        private_train_prompts = [
            instruction
            + "\n".join(
                format_full_datum_for_prompt(field_name, datum) for datum in split
            )
            for split in private_train_splits
        ]
        private_prompts = [
            train_prompt + "\n" + stringified_test_datum
            for train_prompt in private_train_prompts
        ]
        private_prompts_tokens = [
            lm.encoding.encode(prompt) for prompt in private_prompts
        ]

    cnt = 0
    for _ in tqdm.tqdm(range(max_tokens), total=float("inf"), unit=" tokens generated"):
        private_next_logprobs: list[dict[int, float]]
        public_next_logprobs: dict[int, float]
        # Split training dataset
        if subsample_per_token:
            private_train_subset = cast(
                Iterable[dict[str, str]],
                train_subset.shuffle(gen_seed + cnt, keep_in_memory=True).select(
                    range(num_private_train), keep_in_memory=True
                ),
            )
            cnt += 1
            private_train_splits = [
                list(it)
                for it in more_itertools.distribute(
                    num_private_train_splits, private_train_subset
                )
            ]
            # Turn the data into prompts
            private_train_prompts = [
                instruction
                + "\n".join(
                    format_full_datum_for_prompt(field_name, datum) for datum in split
                )
                for split in private_train_splits
            ]
            private_prompts = [
                train_prompt + "\n" + stringified_test_datum
                for train_prompt in private_train_prompts
            ]
            private_prompts_tokens = [
                lm.encoding.encode(prompt) for prompt in private_prompts
            ]
        if no_public_token:
            private_next_logprobs = await asyncio.gather(
                *(
                    next_logprobs(lm, prompt + generated_token_ids, top_p=top_p)
                    for prompt in private_prompts_tokens
                )
            )
            merged_next_probs = merge_logprobs_topk_mean(
                private_next_logprobs,
                None,
                lm.encoding.n_vocab,
                no_public_token,
                normalize_max,
            )

            if normalize_max:
                # scale = 1/lambda
                noise = noise_rng.exponential(scale=sigma, size=lm.encoding.n_vocab)
            else:
                noise = noise_rng.normal(0, sigma, size=lm.encoding.n_vocab)
            merged_next_probs += noise
        else:
            public_next_logprobs = await next_logprobs(
                lm, public_prompt_tokens + generated_token_ids, top_p=top_p
            )
            private_next_logprobs = await asyncio.gather(
                *(
                    normalized_logprobs_for_chosen_tokens(
                        lm,
                        prompt + generated_token_ids,
                        public_next_logprobs.keys(),
                        top_p=top_p,
                    )
                    for prompt in private_prompts_tokens
                )
            )
            merged_next_probs = merge_logprobs_topk_mean(
                private_next_logprobs,
                public_next_logprobs,
                lm.encoding.n_vocab,
                no_public_token,
                normalize_max,
            )
            if normalize_max:
                # scale = 1/lambda
                noise = noise_rng.exponential(
                    scale=sigma, size=len(public_next_logprobs)
                )
            else:
                noise = noise_rng.normal(0, sigma, size=len(public_next_logprobs))
            merged_next_probs[list(public_next_logprobs.keys())] += noise

        next_token_id = int(np.argmax(merged_next_probs))

        if next_token_id in stop_tokens:
            break

        generated_token_ids.append(next_token_id)

        del next_token_id
    return generated_token_ids


async def generate_with_public_prompt(
    public_train_prompt: str,
    stop_tokens: Set[str],
    test_input: str,
    lm: LM,
    field_name,
    max_tokens: int = 500,
) -> list[int]:
    public_prompt = public_train_prompt + format_test_input_for_prompt(
        field_name, test_input
    )
    public_prompt_tokens = lm.encoding.encode(public_prompt)
    public_prompt_tokens = public_prompt

    [completion] = await lm.completions(
        public_prompt_tokens,
        CompletionsSettings(
            temperature=0.0, max_tokens=max_tokens, n=1, stop=list(stop_tokens)
        ),
    )
    generated_tokens = [st.token.token_id for st in completion]
    return generated_tokens


def _main(
    sigma: Annotated[float, typer.Option()],  # noise parameters
    openai_model: Annotated[str, typer.Option()] = "babbage",
    print_prompts: Annotated[bool, typer.Option()] = False,
    # num_private_train=MN. MN=0 with num_valid=4 will get epsilon=0 (4-shot) results.
    num_private_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PRIVATE_TRAIN,
    # by default set to 0. set_num_public_train >0 indicates additional public data available.
    set_num_public_train: Annotated[int, typer.Option()] = DEFAULT_NUM_PUBLIC_TRAIN,
    # num_valid=n. n samples to be generated for n-shot ICL
    num_valid: Annotated[int, typer.Option()] = DEFAULT_NUM_VALID,
    # num_private_train_splits=M
    num_private_train_splits: Annotated[
        int, typer.Option()
    ] = DEFAULT_NUM_PRIVATE_TRAIN_SPLITS,
    num_test: Annotated[int, typer.Option()] = DEFAULT_NUM_TEST,
    # no_public_token=True, RVP=False; no_public_token=False, RVP=True
    no_public_token: Annotated[bool, typer.Option()] = False,
    # subsample_per_token=True: at each token generation, subsample a new test set
    subsample_per_token: Annotated[bool, typer.Option()] = False,
    use_dp_prompts: Annotated[bool, typer.Option()] = False,
    # normalize_max=True, Exponential mechanism; normalize_max=False, Gaussian mechanism
    normalize_max: Annotated[bool, typer.Option()] = False,
    # max_token_per_text=T_max
    max_token_per_text: Annotated[int, typer.Option()] = 20,
    # consistent with default parameters in the documentation https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    top_p: Annotated[float, typer.Option()] = 1,
    # random seed for subsampling in generation
    synth_seed: Annotated[int, typer.Option()] = 0,
    # random seed for n-shot demonstrations sampling in evaluation
    eval_seed: Annotated[int, typer.Option()] = 0,
    # choice bewteen ["Genre", "Director"]
    field_name: Annotated[str, typer.Option()] = "Genre",
    data_path: Annotated[str, typer.Option()] = "./../../data/movie",
):
    async def main():
        if (num_private_train == 0) != (num_private_train_splits == 0):
            raise ValueError(
                "Either both or neither of --num-private-train and --num-private-train-splits can be 0"
            )
        assert field_name in [
            "Director",
            "Genre",
        ]  # field_name from movie dataset include "Actor", "Award", "Character_Name", "Director", "Genre", "Opinion", "Origin", "Plot", "Quote", "Relationship", "Soundtrack", "Year"]
        command = ["python", sys.argv[0]]
        for x in sys.argv[1:]:
            if x.startswith("--"):
                assert '"' not in x and "'" not in x
                command.append(x)
            else:
                assert "'" not in x
                if re.match("^[a-zA-Z0-9_]+$", x):
                    command.append("%s" % x)
                else:
                    command.append("'%s'" % x)
        command = " ".join(command)
        print(command)

        if no_public_token:
            num_public_train = 0
        else:
            num_public_train = set_num_public_train

        lm = api_openai_com(openai_model)
        noise_rng = np.random.RandomState()

        data_files = {"train": "train.csv", "test": "test.csv"}
        data = cast(
            DatasetDict,
            load_dataset(f"{data_path}/{field_name}", data_files=data_files),
        )

        trainset = data["train"].shuffle(seed=synth_seed, keep_in_memory=True)
        print("trainset length", len(trainset))
        if num_public_train > 0:
            public_train_subset = cast(
                Iterable[dict[str, str]],
                trainset.select(
                    range(
                        len(trainset) - num_public_train,
                        len(trainset),
                        keep_in_memory=True,
                    )
                ),
            )
        else:
            public_train_subset = []

        trainset = trainset.select(
            range(len(trainset) - num_public_train), keep_in_memory=True
        )
        query_subset = (
            data["train"]
            .shuffle(seed=eval_seed, keep_in_memory=True)
            .select(range(num_valid), keep_in_memory=True)
        )

        if use_dp_prompts:
            synthetic_examples = []

            # Turn the data into prompts
            instruction = f"Given a propety of {field_name} for the film, generate a description accordingly and make sure to include the given {field_name} in the description.\n\n"
            print(instruction)

            public_train_prompt = instruction + "\n".join(
                format_full_datum_for_prompt(field_name, datum)
                for datum in public_train_subset
            )

            if print_prompts:
                print(public_train_prompt)
                print("=========")

            if normalize_max:
                print("Exponential Mechanism")
                assert num_private_train == 0 or sigma > 0
                if num_private_train > 0:
                    # scale == sigma_calib == 1/lambda. lambda for exponential distribution.
                    sigma_calib = (2 / num_private_train_splits) * (1 / sigma)
            else:
                print("Gaussian Mechanism")
                if num_private_train_splits > 0:
                    sigma_calib = math.sqrt(2) / num_private_train_splits * sigma
                else:
                    sigma_calib = 0
            print(
                f"sigma in command {sigma}. sigma added according to sensitivity {sigma_calib}"
            )

            stop_tokens = {"\n", "<|endoftext|>", " END"}
            stop_tokens_ids = {lm.encoding.encode_single_token(t) for t in stop_tokens}

            client_session.set(aiohttp.ClientSession())

            async with client_session.get():
                for i, test_datum in enumerate(query_subset, 1):
                    print(f"# Example {i}")
                    print(f'{field_name}: "{test_datum["label"]}"')

                    np.random.seed(synth_seed + i)
                    gen_seed = np.random.randint(100000)
                    print(f"gen-seed: {gen_seed}")

                    if num_private_train_splits > 0:
                        generated_token_ids = await generate_with_private_prompts(
                            trainset,
                            num_private_train,
                            num_private_train_splits,
                            instruction,
                            public_train_prompt,
                            stop_tokens_ids,
                            test_datum["label"],
                            lm,
                            noise_rng,
                            sigma_calib,
                            field_name,
                            top_p,
                            no_public_token,
                            subsample_per_token,
                            gen_seed,
                            max_tokens=max_token_per_text
                            - 1,  # need one token length for EOS.
                            normalize_max=normalize_max,
                        )
                    else:
                        generated_token_ids = await generate_with_public_prompt(
                            public_train_prompt,
                            stop_tokens,
                            test_datum["label"],
                            lm,
                            field_name,
                            max_tokens=max_token_per_text,
                        )

                    generated = lm.encoding.decode(generated_token_ids).rstrip('"')

                    print(f"Generated: {generated}\n")
                    output_datum = {}
                    output_datum["content"] = generated.strip()
                    output_datum["label"] = test_datum["label"]
                    synthetic_examples.append(output_datum)

        if num_test > 0 and num_test <= len(data["test"]):
            test_subset = (
                data["test"]
                .shuffle(seed=12345, keep_in_memory=True)
                .select(range(num_test), keep_in_memory=True)
            )
        else:
            test_subset = data["test"]

        all_raw_answers_wout_DP = get_model_response(
            query_subset, test_subset, openai_model, field_name
        )
        all_orig_ans = []
        for resp in all_raw_answers_wout_DP:
            all_orig_ans.append(resp["text"])
        all_orig_ans = [ans.strip() for ans in all_orig_ans]
        test_labels = test_subset["label"]
        orig_accuracy = em_accuracy_helper(all_orig_ans, test_labels)
        print(f"Accuracy (original) without DP: {orig_accuracy}")

        if use_dp_prompts:
            all_raw_answers_w_DP = get_model_response(
                synthetic_examples, test_subset, openai_model, field_name
            )
            all_orig_ans = []
            for resp in all_raw_answers_w_DP:
                all_orig_ans.append(resp["text"])
            all_orig_ans = [ans.strip() for ans in all_orig_ans]
            test_labels = test_subset["label"]
            orig_accuracy = em_accuracy_helper(all_orig_ans, test_labels)
            print(f"Accuracy (original) with DP: {orig_accuracy}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    typer.run(_main)
