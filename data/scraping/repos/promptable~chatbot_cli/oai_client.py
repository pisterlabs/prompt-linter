"""OpenAI Client Wrapper.

This class wraps the OpenAI API and provides a few convenience methods.

Caching is useful for speeding up development and saving money :)

- Caching Completions
- Session management
- Retries and backoff
- Rate limiting
- Response postprocessing
"""
import argparse
import logging
import pprint
import re
import time
from typing import Dict, List, Union

import diskcache
import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INSERT_API_TOKEN = "[insert]"


def postprocess_completion_response(response: Dict) -> Dict:
    """Postprocess OAI completion API response.
    
    Standardize the response format and add additional fields.
    Lets us add new LLMs without changing the API.
    """
    return {
        "response": response,
        "num_tokens": response["usage"]["total_tokens"],
        "all_answers_text": [a["text"] for a in response["choices"]],
        "top_answer_text": response["choices"][0]["text"],
        "latency": response["latency"],
        "usage": response["usage"],
    }


class OAIClient:
    def __init__(
        self,
        api_key: str,
        organization_id: str = None,
        cache: Union[diskcache.Cache, None] = None,
    ):
        self._disk_cache = cache
        openai.organization = organization_id
        openai.api_key = api_key

    def _get_cache_key(self, params: dict) -> str:
        """Get cache key for given parameters.

        Args:
            params (dict): Keyword arguments to pass to `openai.Completion.create()`.
            request_key (str): Key to use for cache lookup and logging.

        Returns:
            str. Cache key.
        """
        return f"completion:" + ":".join(
            [f"{k}={v}" for k, v in sorted(params.items())]
        )

    def _completion_api_call(self, params: dict) -> Dict:
        """Wrapper so we can time the API call w/o cache."""
        logging.debug(f"Calling API with params: {params}")
        start = time.time()
        response: Dict = openai.Completion.create(**params)  # type:ignore
        response["latency"] = round(time.time() - start, 3)
        return response

    def _complete_with_cache(
        self, params: dict, request_tag: Union[str, None] = None
    ) -> Dict:
        """Call Completion API with caching.

        Args:
            params (dict): See `openai.Completion` documentation.
            request_tag (Union[str, None], optional): Tag for easier request debugging/logging.

        Returns:
            Dict: OAI Completion API response.
        """
        cache_key = self._get_cache_key(params)

        logging.debug(f"[OAI:{request_tag}] Prompt:\n{params['prompt']}")

        if self._disk_cache is not None:
            cached_response: Dict = self._disk_cache.get(cache_key)  # type: ignore
            if cached_response is not None:
                logging.info(
                    f"[OAI:{request_tag}] Cache hit!. Entries {len(self._disk_cache)}"
                )
                return cached_response

        response = self._completion_api_call(params)

        logging.debug(f"[OAI:{request_tag}] Latency: {response['latency']}.")

        if self._disk_cache is not None:
            self._disk_cache.set(cache_key, response, tag=request_tag)

        return response

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(
            (
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.Timeout,
                openai.error.TryAgain,
            )
        ),
        before_sleep=before_sleep_log(logger, log_level=logging.INFO),
    )
    def complete(
        self,
        prompt: str,
        stop: Union[List[str], None] = None,
        n: int = 1,
        best_of: int = 1,
        top_p: int = 1,
        temperature: float = 0,
        logprobs: int | None = None,
        max_tokens: int = 256,
        frequency_penalty: int = 0,
        presence_penalty: int = 0,
        model: str = "text-davinci-002",
        logit_bias: Union[Dict[str, float], None] = None,
        request_tag: Union[str, None] = None,
        mode: str = "complete",  # or insert
    ) -> Dict:
        """Call OpenAI Completion API.

        See https://beta.openai.com/docs/api-reference/completions for param descriptions.

        Args:
            prompt (str): Prompt to complete.
            request_tag (str): Request Tag to use for cache lookup and logging.
            suffix (str): Appended to prompt for INSERT requests. Example:
                complete(
                    model="text-davinci-002",
                    prompt="We're writing to ",
                    suffix=" this into a paragraph.",
                )
        Returns:
            Dict. Response from OpenAI Completion API.
        """
        logging.debug(f"[OAI] Prompt:\n{prompt}")

        suffix = None
        if mode == "insert":
            if prompt.lower().count(INSERT_API_TOKEN) != 1:
                raise ValueError(
                    f"Prompt must contain exactly 1 instance of '{INSERT_API_TOKEN}' token."
                )
            prompt, suffix = re.split(r"\[insert\]", prompt, flags=re.IGNORECASE)

        params = dict(
            prompt=prompt,
            model=model,
            n=n,
            top_p=top_p,
            best_of=best_of,  # we always return all answers so best_of = n
            temperature=temperature,
            logprobs=logprobs,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logit_bias=logit_bias or {},
            suffix=suffix,
        )

        logging.debug(f"[OAI:{request_tag}] Params: {params}")

        response = self._complete_with_cache(params, request_tag)

        logging.debug(f"[OAI] Response:\n{response}")

        result = postprocess_completion_response(response)
        result["request_params"] = params
        result["request_tag"] = request_tag

        logging.debug(f"[OAI] Post-processed:\n{result}")

        return result


if __name__ == "__main__":
    """Example Usage of OAIClient.
    
    Usage:
        python oai_client.py --prompt "Hello, how are you?" --model "text-davinci-002"
    """
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="text-davinci-002")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    
    from settings import Settings
    cfg = Settings.from_env_file()

    cache = None
    if args.cache_dir is not None:
        cache = diskcache.Cache(args.cache_dir)

    oai_client = OAIClient(
        api_key=cfg.openai_api_key,
        organization_id=cfg.openai_org_id,
        cache=cache,
    )

    result = oai_client.complete(prompt=args.prompt, request_tag="test")

    pprint.pp(result)
