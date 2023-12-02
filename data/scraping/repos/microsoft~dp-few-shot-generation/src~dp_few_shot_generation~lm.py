# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections.abc import Sequence, Set

from lmapi.async_tools import limits
from lmapi.auth import OpenAiApiKey
from lmapi.lm import LM, CompletionsSettings
from lmapi.openai import OpenAI

from dp_few_shot_generation.prob_utils import log_normalize


def api_openai_com(model_name: str) -> OpenAI:
    return OpenAI.create(
        "https://api.openai.com/v1/completions",
        # This list is taken from
        # https://github.com/openai/tiktoken/blob/095924e02c85617df6889698d94515f91666c7ea/tiktoken/model.py#L13-L53
        # and modified, currently to accommodate how text-davinci-003 can actually produce <|fim_...|> tokens.
        {
            # chat
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            # text
            "text-davinci-003": "p50k_edit",
            "text-davinci-002": "p50k_base",
            "text-davinci-001": "r50k_base",
            "text-curie-001": "r50k_base",
            "text-babbage-001": "r50k_base",
            "text-ada-001": "r50k_base",
            "davinci": "r50k_base",
            "curie": "r50k_base",
            "babbage": "r50k_base",
            "ada": "r50k_base",
            # code
            "code-davinci-002": "p50k_base",
            "code-davinci-001": "p50k_base",
            "code-cushman-002": "p50k_base",
            "code-cushman-001": "p50k_base",
            "davinci-codex": "p50k_base",
            "cushman-codex": "p50k_base",
            # edit
            "text-davinci-edit-001": "p50k_edit",
            "code-davinci-edit-001": "p50k_edit",
            # embeddings
            "text-embedding-ada-002": "cl100k_base",
            # old embeddings
            "text-similarity-davinci-001": "r50k_base",
            "text-similarity-curie-001": "r50k_base",
            "text-similarity-babbage-001": "r50k_base",
            "text-similarity-ada-001": "r50k_base",
            "text-search-davinci-doc-001": "r50k_base",
            "text-search-curie-doc-001": "r50k_base",
            "text-search-babbage-doc-001": "r50k_base",
            "text-search-ada-doc-001": "r50k_base",
            "code-search-babbage-code-001": "r50k_base",
            "code-search-ada-code-001": "r50k_base",
            # open source
            "gpt2": "gpt2",
        }[model_name],
        OpenAiApiKey(os.environ["OPENAI_API_KEY"]),
        limits.AdaptiveLimiter(),
        {"model": model_name},
    )


MAX_TOP_LOGPROBS = 100
MAX_LOGIT_BIAS = 100
MIN_LOGIT_BIAS = -100


async def next_logprobs(
    self: LM, prompt: str | Sequence[int], top_p=1
) -> dict[int, float]:
    # TODO: Don't hardcode "100" here
    [sampled_tokens] = await self.completions(
        prompt,
        CompletionsSettings(n=1, max_tokens=1, logprobs=100, stop=["<test_for_stop>"]),
    )
    if len(sampled_tokens) == 0:
        if isinstance(prompt, str):
            prompt += "<|endoftext|>"
        else:
            prompt = [*prompt, self.encoding.encode_single_token("<|endoftext|>")]
        [[*_prev_tokens, sampled_token]] = await self.completions(
            prompt,
            CompletionsSettings(
                n=1, max_tokens=0, logprobs=100, echo=True, top_p=top_p
            ),
        )
    else:
        [sampled_token] = sampled_tokens

    return {tlp.token_id: tlp.logprob for tlp in sampled_token.top_choices}


async def normalized_logprobs_for_chosen_tokens(
    self: LM, prompt: Sequence[int], chosen_tokens: Set[int], top_p: float
) -> dict[int, float]:
    """Compute the probability that the prompt will be continued with each of the chosen tokens.

    The returned probability distribution is normalized over just the chosen tokens."""

    assert (
        len(chosen_tokens) <= MAX_TOP_LOGPROBS
    ), f"chosen_tokens must be <= {MAX_TOP_LOGPROBS} in length"

    logit_bias = {token_id: MAX_LOGIT_BIAS for token_id in chosen_tokens}
    [sampled_tokens] = await self.completions(
        prompt,
        CompletionsSettings(
            n=1,
            max_tokens=1,
            logprobs=MAX_TOP_LOGPROBS,
            logit_bias=logit_bias,
            top_p=top_p,
        ),
    )
    if len(sampled_tokens) == 0:
        # Fall back to querying over the set
        chosen_tokens_list = list(chosen_tokens)

        result = await self.completions(
            [[*prompt, token_id] for token_id in chosen_tokens_list],
            CompletionsSettings(n=1, max_tokens=0, logprobs=0, echo=True),
        )
        unnormalized_logprobs = {
            sampled_token.token.token_id: sampled_token.token.logprob
            for [*_prev_tokens, sampled_token] in result
        }
        return log_normalize(unnormalized_logprobs)
    else:
        [sampled_token] = sampled_tokens
        biased_logprobs = {
            tlp.token_id: tlp.logprob for tlp in sampled_token.top_choices
        }
        biased_logprobs_for_tokens = {
            token_id: biased_logprobs.get(token_id, float("-inf"))
            for token_id in chosen_tokens
        }
        return log_normalize(biased_logprobs_for_tokens)
