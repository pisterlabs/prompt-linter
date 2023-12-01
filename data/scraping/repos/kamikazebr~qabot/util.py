import json
import time
from functools import wraps
from logging.config import dictConfig

from langchain.schema import LLMResult
from pydantic import BaseModel

import aiohttp

from logger.hivemind_logger import logger


class Timeless:
    def __init__(self):
        self.start = time.time()

    def end(self, thing: str | None = None):
        end = time.time() - self.start
        # convert to H:M:S
        m, s = divmod(end, 60)
        h, m = divmod(m, 60)
        # get name of function too
        print(f"{f'{thing} ' if thing else ''}took {h:.0f}h:{m:.0f}m:{s:.0f}s")


def time_start() -> Timeless:
    return Timeless()


def atimeit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time() - start
        # convert to H:M:S
        m, s = divmod(end, 60)
        h, m = divmod(m, 60)
        # get name of function too
        logger.debug(f"{func.__name__} took {h:.0f}h:{m:.0f}m:{s:.0f}s")
        return result

    return wrapper


# create a decorator that print time of execution of async function using time.time() in H:M:S
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        # convert to H:M:S
        m, s = divmod(end, 60)
        h, m = divmod(m, 60)
        # get name of function too
        print(f"{func.__name__} took {h:.0f}h:{m:.0f}m:{s:.0f}s")
        return result

    return wrapper


def get_total_tokens(llm_result: LLMResult) -> int:
    token_usage = llm_result.llm_output["token_usage"]
    total_tokens = token_usage["total_tokens"]
    return total_tokens


def create_deeplake():
    pass


class Settings(BaseModel):
    ENVIRONMENT: str = 'local'


settings = Settings()


def configure_logging() -> None:
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {  # correlation ID filter must be added here to make the %(correlation_id)s formatter work
                'correlation_id': {
                    '()': 'asgi_correlation_id.CorrelationIdFilter',
                    'uuid_length': 8 if not settings.ENVIRONMENT == 'local' else 32,
                    'default_value': '-',
                },
            },
            'formatters': {
                'console': {
                    'class': 'logging.Formatter',
                    'datefmt': '%H:%M:%S',
                    # formatter decides how our console logs look, and what info is included.
                    # adding %(correlation_id)s to this format is what make correlation IDs appear in our logs
                    'format': '%(levelname)s:\t\b%(asctime)s %(name)s:%(lineno)d [%(correlation_id)s] %(message)s - %(extras)s',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    # Filter must be declared in the handler, otherwise it won't be included
                    'filters': ['correlation_id'],
                    'formatter': 'console',
                },
            },
            # Loggers can be specified to set the log-level to log, and which handlers to use
            'loggers': {
                # project logger
                'app': {'handlers': ['console'], 'level': 'DEBUG', 'propagate': True},
                # 'hivemind': {'handlers': ['console'], 'level': 'DEBUG', 'propagate': True},
                # third-party package loggers
                'databases': {'handlers': ['console'], 'level': 'WARNING'},
                'httpx': {'handlers': ['console'], 'level': 'INFO'},
                'asgi_correlation_id': {'handlers': ['console'], 'level': 'WARNING'},
            },
        }
    )


async def async_post_request(url, json):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json) as response:
            return await response.json()


async def async_get_request(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Error while making the request: {await response.text()}")
                    # Handle HTTP errors here, if necessary
                    return None
    except aiohttp.ClientError as e:
        # Handle specific aiohttp exceptions, if necessary
        logger.error(f"Error while making the request: {e}")
        return None
    except Exception as e:
        # Handle other non-specific exceptions
        logger.error(f"Unexpected error: {e}")
        return None
