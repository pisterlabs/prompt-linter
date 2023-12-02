import concurrent.futures
import logging
from typing import List


class Filter:
    def safe_check(self, input: str, timeout_seconds: int = 5):
        """If moderation request takes longer than timeout, simply allow it."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.check, input)
            try:
                output = future.result(timeout=timeout_seconds)
                return output
            except concurrent.futures.TimeoutError:
                logging.warning("OpenAI filter timed out. Allowing request.")
                return {
                    "response": None,
                    "any_flagged": False,
                }

    def check(self, input: str | List[str]):
        raise NotImplementedError


import openai


class OpenAIFilter(Filter):
    def __init__(self, api_key):
        self.api_key = api_key

    def check(self, input: str | List[str]):
        response = openai.Moderation.create(input=input, api_key=self.api_key)
        return {
            "response": response,
            # Results include a flagged boolean for each input, look at all of them
            "any_flagged": any([result["flagged"] for result in response["results"]]),
        }
