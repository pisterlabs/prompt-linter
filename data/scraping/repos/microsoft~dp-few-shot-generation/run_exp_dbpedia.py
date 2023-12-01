# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import math
import re
import sys
import time
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
DEFAULT_NUM_PRIVATE_TRAIN_SPLITS = 40
DEFAULT_NUM_TEST = 1000

labels = [
    "Company",
    "School",
    "Artist",
    "Ath",
    "Polit",
    "Transportation",
    "Building",
    "Nature",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Book",
]
label_dict = {
    0: ["Company"],
    1: ["School"],
    2: ["Artist"],
    3: ["Ath"],
    4: ["Polit"],
    5: ["Transportation"],
    6: ["Building"],
    7: ["Nature"],
    8: ["Village"],
    9: ["Animal"],
    10: ["Plant"],
    11: ["Album"],
    12: ["Film"],
    13: ["Book"],
}


def format_full_datum_for_prompt(labels, datum: dict[str, str]):
    return f'Document Type: "{labels[datum["label"]]}"\nText: "{datum["content"] + " END"}"\n'


def format_test_input_for_prompt(labels, test_input: int):
    return f'Document Type: "{labels[test_input]}"\nText: "'


def construct_prompt_same(train_examples, test_example):
    labels_str = ", ".join(labels)
    prompt = (
        f"Classify the documents based on whether they are about a {labels_str}.\n\n"
    )
    for train_example in train_examples:
        prompt += "Article: " + train_example["content"] + "\n"
        prompt += "Answer: " + label_dict[train_example["label"]][0] + "\n\n"
    prompt += "Article: " + test_example["content"] + "\n"
    prompt += "Answer:"
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


def get_model_response(data, test_examples, openai_model):
    all_raw_answers = []

    prompts = []
    train_examples = data

    for test_example in test_examples:
        prompts.append(construct_prompt_same(train_examples, test_example))

    chunked_prompts = list(chunks(prompts, 20))
    for test_chunk in chunked_prompts:
        response = complete(test_chunk, l=1, model_name=openai_model, num_log_probs=100)

        for answer_id, answer in enumerate(response["choices"]):
            all_raw_answers.append(answer)

    return all_raw_answers


def get_label_probs(all_raw_answers, test_subset):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(label_dict)
    approx = False
    assert len(all_raw_answers) == len(test_subset)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    cnt = 0
    for i, ans in enumerate(all_raw_answers):
        try:
            top_logprobs = ans["logprobs"]["top_logprobs"][
                0
            ]  # [0] since we only ask for complete one more token
        except:
            cnt += 1  # cnt for corner case
        label_probs = [0] * len(label_dict.keys())
        for j, label_list in label_dict.items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j)  # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs)  # prob not normalized

    return all_label_probs  # NOT NORMALIZED


def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        if np.sum(label_probs) > 0:  # corner case np.sum(label_probs)=0.
            label_probs = label_probs / np.sum(label_probs)  # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)


def get_p_content_free(train_subset, openai_model, content_free_inputs=("N/A",)):
    """Query model with content free input, return its prediction probability for each label"""
    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt_same(train_subset, content_free_input)
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(
                    complete(
                        prompt + " " + a, 0, openai_model, echo=True, num_log_probs=1
                    )["choices"][0]["logprobs"]["token_logprobs"][-1]
                )
            p_y[i] = prob
        all_p_y.append(p_y)
    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)  # normalize
    return p_y


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
    test_input: int,
    lm: LM,
    noise_rng: np.random.RandomState,
    sigma: float,
    labels,
    top_p,
    no_public_token: bool,
    subsample_per_token: bool,
    sample_same_label_prompts: bool,
    gen_seed: int,
    max_tokens: int,
    normalize_max: bool = False,
) -> list[int]:
    generated_token_ids: list[int] = []

    stringified_test_datum = format_test_input_for_prompt(labels, test_input)
    public_prompt = public_train_prompt + stringified_test_datum
    public_prompt_tokens = lm.encoding.encode(public_prompt)

    assert num_private_train_splits > 0
    if sample_same_label_prompts:
        select_list = []
        for i in range(len(trainset)):
            if trainset[i]["label"] == test_input:
                select_list.append(i)
        train_subset = trainset.select(select_list, keep_in_memory=True)
    else:
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
            + "\n".join(format_full_datum_for_prompt(labels, datum) for datum in split)
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
                    format_full_datum_for_prompt(labels, datum) for datum in split
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
    labels,
    max_tokens: int = 500,
) -> list[int]:
    public_prompt = public_train_prompt + format_test_input_for_prompt(
        labels, test_input
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


def select_uniform_n_shots_over_labels(data, n_shots):
    select_list = []
    n_shots_per_label = math.ceil(n_shots / len(labels))
    labels_counter = {label[1][0]: n_shots_per_label for label in label_dict.items()}
    n_shots_selected = 0
    for i in range(len(data)):
        label = label_dict[data[i]["label"]][0]
        if labels_counter[label] == 0:
            continue
        else:
            labels_counter[label] -= 1
            select_list.append(i)
            n_shots_selected += 1
        if n_shots_selected == n_shots:
            break
    query_subset = data.select(select_list, keep_in_memory=True)
    return query_subset


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
    # sample_same_label_prompts=True: sample subsets from the sets with same targeted labels.
    sample_same_label_prompts: Annotated[bool, typer.Option()] = False,
    # normalize_max=True, Exponential mechanism; normalize_max=False, Gaussian mechanism
    normalize_max: Annotated[bool, typer.Option()] = False,
    # max_token_per_text=T_max
    max_token_per_text: Annotated[int, typer.Option()] = 100,
    # consistent with default parameters in the documentation https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#completions
    top_p: Annotated[float, typer.Option()] = 1,
    # random seed for subsampling in generation
    synth_seed: Annotated[int, typer.Option()] = 0,
    # random seed for n-shot demonstrations sampling in evaluation
    eval_seed: Annotated[int, typer.Option()] = 0,
):
    async def main():
        if (num_private_train == 0) != (num_private_train_splits == 0):
            raise ValueError(
                "Either both or neither of --num-private-train and --num-private-train-splits can be 0"
            )
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

        data = cast(DatasetDict, load_dataset("dbpedia_14"))
        print(labels)

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
        queryset = data["train"].shuffle(seed=eval_seed, keep_in_memory=True)
        query_subset = select_uniform_n_shots_over_labels(queryset, num_valid)

        if use_dp_prompts:
            synthetic_examples = []

            # Turn the data into prompts
            instruction = "Given a label of document type, generate the chosen type of document accordingly.\n\n"

            public_train_prompt = instruction + "\n".join(
                format_full_datum_for_prompt(labels, datum)
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
                    print(f'Document Type: "{labels[test_datum["label"]]}"')
                    print(f'References:\n "{test_datum["content"]}"')

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
                            labels,
                            top_p,
                            no_public_token,
                            subsample_per_token,
                            sample_same_label_prompts,
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
                            labels,
                            max_tokens=max_token_per_text,
                        )

                    generated = lm.encoding.decode(generated_token_ids).rstrip('"')

                    print(f"Generated: {generated}\n")
                    output_datum = {}
                    output_datum["content"] = generated.strip()
                    output_datum["label"] = test_datum["label"]
                    synthetic_examples.append(output_datum)

        if num_test > 0:
            test_subset = (
                data["test"]
                .shuffle(seed=12345, keep_in_memory=True)
                .select(range(num_test), keep_in_memory=True)
            )
            test_labels = [test_example["label"] for test_example in test_subset]

            content_free_inputs = [
                {"content": "N/A"},
                {"content": ""},
                {"content": "[MASK]"},
            ]
            p_cf_wout_DP = get_p_content_free(
                query_subset, openai_model, content_free_inputs=content_free_inputs
            )

            all_raw_answers_wout_DP = get_model_response(
                query_subset, test_subset, openai_model
            )
            all_label_probs_wout_DP = get_label_probs(
                all_raw_answers_wout_DP, test_subset
            )

            acc_original_wout_DP = eval_accuracy(all_label_probs_wout_DP, test_labels)
            acc_calibrated_wout_DP = eval_accuracy(
                all_label_probs_wout_DP,
                test_labels,
                mode="diagonal_W",
                p_cf=p_cf_wout_DP,
            )

            print(f"Accuracy (original) without DP: {acc_original_wout_DP}")
            print(f"Accuracy (calibrated) without DP: {acc_calibrated_wout_DP}")

            if use_dp_prompts:
                p_cf_w_DP = get_p_content_free(
                    synthetic_examples,
                    openai_model,
                    content_free_inputs=content_free_inputs,
                )
                all_raw_answers_w_DP = get_model_response(
                    synthetic_examples, test_subset, openai_model
                )

                all_label_probs_w_DP = get_label_probs(
                    all_raw_answers_w_DP, test_subset
                )

                acc_original_w_DP = eval_accuracy(all_label_probs_w_DP, test_labels)
                acc_calibrated_w_DP = eval_accuracy(
                    all_label_probs_w_DP, test_labels, mode="diagonal_W", p_cf=p_cf_w_DP
                )

                print(f"Accuracy (original) with DP: {acc_original_w_DP}")
                print(f"Accuracy (calibrated) with DP: {acc_calibrated_w_DP}")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    typer.run(_main)
