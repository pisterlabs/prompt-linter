from __future__ import annotations

import os
import time

import openai

from miniboss.config import Config
from miniboss.llm.modelsinfo import COSTS
from miniboss.logs import logger
from miniboss.singleton import Singleton


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0

    import os
    import time

    def create_chat_completion(
        self,
        messages: list,  # type: ignore
        model: str | None = None,
        temperature: float = None,
        max_tokens: int | None = None,
        deployment_id=None,
    ) -> str:
        """
        Create a chat completion and update the cost.
        Args:
        messages (list): The list of messages to send to the API.
        model (str): The model to use for the API call.
        temperature (float): The temperature to use for the API call.
        max_tokens (int): The maximum number of tokens for the API call.
        Returns:
        str: The AI's response.
        """
        cfg = Config()
        if temperature is None:
            temperature = cfg.temperature

        MAX_RETRIES = 5
        BACKOFF_START = 1  # initial backoff delay in seconds
        fallback_model = "gpt-3.5-turbo"  # model to use if all retries fail

        for attempt in range(MAX_RETRIES):
            try:
                if deployment_id is not None:
                    response = openai.ChatCompletion.create(
                        deployment_id=deployment_id,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=cfg.openai_api_key,
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=cfg.openai_api_key,
                    )
                logger.debug(f"Response: {response}")
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                self.update_cost(prompt_tokens, completion_tokens, model)
                return response

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:  # if not the last attempt
                    delay = BACKOFF_START * 2**attempt
                    logger.info(f"Waiting for {delay} seconds before retrying...")
                    time.sleep(delay)
                elif model == self.smart_llm_model:  # if last attempt and using gpt-4
                    logger.info("Retries exhausted. Falling back to gpt-3.5-turbo.")
                    model = fallback_model
                else:  # if last attempt and already using fallback model
                    raise

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += (
            prompt_tokens * COSTS[model]["prompt"]
            + completion_tokens * COSTS[model]["completion"]
        ) / 1000
        logger.debug(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        total_budget (float): The total budget for API calls.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget
