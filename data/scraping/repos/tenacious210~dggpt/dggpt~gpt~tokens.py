# Calculates various things from openai tokens
import logging
import tiktoken
from dggpt.config import read_monthly_tokens
from .completions import CHAT_MODEL

logger = logging.getLogger(__name__)

encoding = tiktoken.encoding_for_model(CHAT_MODEL)


def count_tokens(convo: list[dict]) -> int:
    """Count the amount of tokens present in an openai convo"""
    num_tokens = 0
    for message in convo:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens


def get_cost_from_tokens() -> int:
    """Read the token tally from monthly_tokens.json and return it in dollars"""
    return round(read_monthly_tokens() / 1000 * 0.0015, 2)
