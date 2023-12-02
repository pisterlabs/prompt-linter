from __future__ import annotations

from unittest.mock import patch
import functools
import time
import os
import logging


from openai.error import (
    APIError,
    RateLimitError,
    InvalidRequestError,
    APIConnectionError,
    Timeout,
)


def retry_openai_api(
    num_retries: int = 10,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Decorate a function to retry OpenAI API call when it fails.

    This function uses exponential backoff strategy for retries.

    Args:
        num_retries (int, optional): Number of retries. Defaults to 10.
        backoff_base (float, optional): Base for exponential backoff. Defaults to 2.0.
        warn_user (bool, optional): Whether to warn the user. Defaults to True.

    Returns:
        Callable: The decorated function.
    """

    def _wrapper(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError(
                    "OPENAI_API_KEY environment variable must be set when using OpenAI API."
                )
            user_warned = not warn_user
            num_attempts = num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except RateLimitError as e:
                    if attempt == num_attempts:
                        raise

                    logging.error("Error: Reached rate limit, passing...")
                    if not user_warned:
                        logging.warning(
                            """Please double check that you have setup a paid OpenAI API 
                              Account. You can read more here: https://docs.agpt.co/setup/#getting-an-api-key"""
                        )
                        user_warned = True
                except InvalidRequestError as e:
                    if 'not exist' in str(e):
                        logging.warning(f"Requested model does not exist. Using default model gpt-3.5-turbo-0613.")
                        kwargs['model'] = "gpt-3.5-turbo-0613"
                        return func(*args, **kwargs)
                    else:
                        logging.error(f"OpenAI API Invalid Request: {e}")
                        user_warned = True
                except APIConnectionError as e:
                    logging.error(
                        f"OpenAI API Connection Error: {e}"
                    )
                    user_warned = True
                except Timeout as e:
                    logging.error(f"OpenAI API Timeout Error: {e}")
                    user_warned = True
                except APIError as e:
                    if (e.http_status not in [502, 429]) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logging.error(f"Error: API Bad gateway. Waiting {backoff} seconds...")
                time.sleep(backoff)

        return _wrapped

    return _wrapper
