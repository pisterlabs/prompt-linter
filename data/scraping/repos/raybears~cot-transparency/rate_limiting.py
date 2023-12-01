import logging
import threading
import time
from functools import wraps
from typing import Callable, Optional, Protocol, TypeVar

import tiktoken

from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import StrictChatMessage
from cot_transparency.util import setup_logger

exit_event = threading.Event()

rate_limit_logger = setup_logger("rate_limit_logger", logging.INFO)


class LeakyBucketRateLimiter:
    def __init__(
        self,
        tokens_per_minute: int,
        log_every_n_requests: int = 200,
        logger: Optional[logging.Logger] = None,
    ):
        self.tokens_per_minute: float = tokens_per_minute
        self.tokens_available: float = tokens_per_minute
        self.last_request: float = time.time()
        self.last_log: float = time.time()
        self.tokens_used: int = 0
        self.request_counter: int = 0
        self.lock = threading.Lock()
        self.log_every_n_requests = log_every_n_requests
        self.logger = logger

    def consume(self, tokens: int, model: str):
        with self.lock:
            time_elapsed = time.time() - self.last_request
            fill_rate = self.tokens_per_minute / 60
            self.tokens_available += time_elapsed * fill_rate

            if self.tokens_available > self.tokens_per_minute:
                self.tokens_available = self.tokens_per_minute

            while self.tokens_available < tokens:
                if exit_event.is_set():
                    raise Exception("Exiting program early because exit event is set")
                time.sleep(1)
                time_elapsed = time.time() - self.last_request
                self.tokens_available += time_elapsed * fill_rate

            self.tokens_available -= tokens
            self.tokens_used += tokens
            self.last_request = time.time()
            self.request_counter += 1

            if self.logger and self.request_counter >= self.log_every_n_requests:  # N: log every N calls
                actual_rate = self.tokens_used / ((time.time() - self.last_log) / 60)
                self.logger.info(
                    f"Model: {model} In the last {self.log_every_n_requests} requests: used {self.tokens_used} tokens "
                    f"at a rate of {actual_rate:.0f} tokens/minute"
                )
                # Reset tokens used and counter after logging
                self.tokens_used = 0
                self.request_counter = 0
                self.last_log = time.time()


def get_num_tokens(config: OpenaiInferenceConfig, messages: list[StrictChatMessage]):
    completion_tokens = config.max_tokens
    messages_as_dicts = [chat.model_dump() for chat in messages]
    encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    # note: future models may deviate from this
    # chat completions are also in list
    for message in messages_as_dicts:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens + completion_tokens


T = TypeVar("T", covariant=True)


class CallingModelFunction(Protocol[T]):
    # We only can rate limit functions with the following signature:
    def __call__(
        self,
        config: OpenaiInferenceConfig,
        messages: list[StrictChatMessage],
        organization: Optional[str] = None,
    ) -> T:
        ...


def token_rate_limiter(
    tokens_per_minute: int, logger: Optional[logging.Logger] = rate_limit_logger
) -> Callable[[CallingModelFunction[T]], CallingModelFunction[T]]:
    rate_limiter = LeakyBucketRateLimiter(tokens_per_minute, logger=logger)

    def decorator(func: CallingModelFunction[T]) -> CallingModelFunction[T]:
        @wraps(func)
        def wrapper(
            config: OpenaiInferenceConfig,
            messages: list[StrictChatMessage],
            organization: Optional[str] = None,
        ) -> T:
            tokens = get_num_tokens(config=config, messages=messages)
            rate_limiter.consume(tokens, model=config.model)
            return func(config, messages, organization)

        return wrapper

    return decorator
