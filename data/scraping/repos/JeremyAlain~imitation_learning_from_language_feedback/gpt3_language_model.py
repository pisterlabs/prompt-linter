import math
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openai
from openai.openai_object import OpenAIObject
import pandas as pd
from tqdm import tqdm

from utilities.language_model_utilities import (
    list_to_batches_generator,
    set_openai_key_to_environment_variable,
)


class GPT3LanguageModel:
    """Wrapper to call the OpenAI API for GPT3."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        set_openai_key_to_environment_variable()

    def generate_completion(
        self,
        prompts: Union[str, List[str]],
        max_tokens_to_generate: int,
        number_of_log_probabilities: int = 0,
        batch_size: int = 1,
        echo: bool = False,
        temperature: float = 0,
        retry_delay: int = 10,
        stop_words: List[str] = None,
        number_of_generations_per_prompt: int = 1,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Dict[str, int] = None,
        save_intermediate_results_path: str = None,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        number_of_iterations = math.ceil(len(prompts) / batch_size)

        # if save_intermediate_results_path is not None:
        # assert not os.path.isfile(save_intermediate_results_path), save_intermediate_results_path

        all_responses = []
        for prompt_batch in tqdm(
            list_to_batches_generator(list=prompts, batch_size=batch_size),
            total=number_of_iterations,
        ):
            try:
                batch_response = self._call_openai_api(
                    prompts=prompt_batch,
                    max_tokens_to_generate=max_tokens_to_generate,
                    number_of_log_probabilities=number_of_log_probabilities,
                    echo=echo,
                    temperature=temperature,
                    retry_delay=retry_delay,
                    stop_words=stop_words,
                    number_of_generations_per_prompt=number_of_generations_per_prompt,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                )

                if number_of_log_probabilities == 0:
                    response = [choice["text"] for choice in batch_response["choices"]]
                else:
                    response = [
                        {
                            "text": choice["text"],
                            "log_probabilities": dict(choice["logprobs"]),
                        }
                        for choice in batch_response["choices"]
                    ]
                assert (
                    len(response)
                    == len(prompt_batch) * number_of_generations_per_prompt
                )
                all_responses += response
                pd.DataFrame(all_responses).to_json(save_intermediate_results_path)
            except (RuntimeError):
                pd.DataFrame(all_responses).to_json(save_intermediate_results_path)
        return all_responses

    def _call_openai_api(
        self,
        prompts: Union[str, List[str]],
        max_tokens_to_generate: int,
        number_of_log_probabilities: Optional[int],
        echo: bool,
        temperature: float = 0,
        retry_delay: int = 10,
        stop_words: List[str] = None,
        number_of_generations_per_prompt: int = 1,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Dict[str, int] = None,
    ) -> OpenAIObject:
        sleep_time = 0
        while True:
            try:
                if logit_bias is not None:
                    response = openai.Completion.create(
                        engine=self.model_name,
                        prompt=prompts,
                        max_tokens=max_tokens_to_generate,
                        echo=echo,
                        logprobs=number_of_log_probabilities,
                        temperature=temperature,
                        stop=stop_words,
                        n=number_of_generations_per_prompt,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        logit_bias=logit_bias,
                    )
                else:
                    response = openai.Completion.create(
                        engine=self.model_name,
                        prompt=prompts,
                        max_tokens=max_tokens_to_generate,
                        echo=echo,
                        logprobs=number_of_log_probabilities,
                        temperature=temperature,
                        stop=stop_words,
                        n=number_of_generations_per_prompt,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                    )
                break
            except (
                openai.APIError,
                openai.error.RateLimitError,
                openai.error.APIConnectionError,
            ):
                if sleep_time == 0:
                    print("Sleeping...")
                sleep_time += retry_delay
                time.sleep(retry_delay)
        if sleep_time != 0:
            print(f"\tSlept {sleep_time}s")
        return response


class GPT3TextEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_names = [
            "text-similarity-ada-001",
            "text-similarity-babbage-001",
            "text-similarity-curie-001",
            "text-similarity-davinci-001",
        ]
        assert self.model_name in self.model_names
        set_openai_key_to_environment_variable()

    def embed_text(self, text: str) -> np.ndarray:
        assert self.model_name
        sleep_time = 0
        while True:
            try:
                embedding = np.array(
                    openai.Embedding.create(input=[text], engine=self.model_name)[
                        "data"
                    ][0]["embedding"]
                )
                break
            except (
                openai.APIError,
                openai.error.RateLimitError,
                openai.error.APIConnectionError,
            ):
                if sleep_time == 0:
                    print("Sleeping...")
                sleep_time += 10
                time.sleep(10)
        if sleep_time != 0:
            print(f"\tSlept {sleep_time}s")
        return embedding
