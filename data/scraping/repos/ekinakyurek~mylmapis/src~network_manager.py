import asyncio
import logging
from functools import wraps
from math import ceil, log2
from random import random
from openai import APIError
from openai.error import (
    APIConnectionError,
    AuthenticationError,
    InvalidRequestError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
    TryAgain,
)

OPENAI_MAX_RETRY = 10
OPENAI_REFRESH_QUOTA = 60
OPENAI_EXP_CAP = int(ceil(log2(OPENAI_REFRESH_QUOTA)))


def async_retry_with_exp_backoff(task):
    async def wrapper(*args, **kwargs):
        for i in range(OPENAI_MAX_RETRY + 1):
            wait_time = (1 << min(i, OPENAI_EXP_CAP)) + random() / 10
            try:
                return await task(*args, **kwargs)
            except (
                RateLimitError,
                ServiceUnavailableError,
                APIConnectionError,
                APIError,
                TryAgain,
            ) as msg:
                if i == OPENAI_MAX_RETRY:
                    logging.error(
                        "Retry, TooManyRequests or Server Error. %s", str(msg)
                    )
                    raise msg
                else:
                    logging.warning(
                        f"Waiting {round(wait_time, 2)} seconds for API... {msg}",
                    )
                    await asyncio.sleep(wait_time)
            except AuthenticationError as msg:
                # No way to handle
                logging.error("AuthenticationError: %s", str(msg))
                raise Exception(
                    "AuthenticationError: Incorrect API key is provided.",
                ) from msg
            except InvalidRequestError as msg:
                logging.error("InvalidRequestError: %s", str(msg))
                raise msg
            except OpenAIError as msg:
                logging.error("API Request failed. %s", str(msg))
                raise msg
            except Exception as msg:
                logging.error("Error unrelated to API. %s", str(msg))
                raise msg

    return wrapper
