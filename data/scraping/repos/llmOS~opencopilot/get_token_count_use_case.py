from dataclasses import dataclass
from typing import List

from langchain.chat_models.base import BaseChatModel


@dataclass(frozen=True)
class CacheObject:
    text: str
    token_count: int


cache: List[CacheObject] = []


def execute(
    text: str,
    llm: BaseChatModel,
    is_use_cache: bool = False,
) -> int:
    if not is_use_cache:
        return llm.get_num_tokens(text)

    for c in cache:
        if c.text == text:
            return c.token_count
    token_count = llm.get_num_tokens(text)
    cache.append(CacheObject(text=text, token_count=token_count))
    return token_count
