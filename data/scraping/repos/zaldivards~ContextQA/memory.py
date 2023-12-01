from typing import Literal

from langchain.llms import OpenAI
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    RedisChatMessageHistory,
)
from langchain.schema import BaseMemory

from contextqa import settings

envs = settings()
_PROMPT_KEYS = {
    "default": {"input_key": "input", "memory_key": "history"},
    "defaultv2": {"input_key": "input", "memory_key": "chat_history"},
    "context": {"input_key": "question", "memory_key": "chat_history"},
}


def _prompt_keys(kind: str, internet_access: bool = False) -> dict[str, str]:
    if internet_access:
        return _PROMPT_KEYS["defaultv2"]
    return _PROMPT_KEYS[kind]


def _requires_raw(session: str, internet_access: bool) -> bool:
    return session != "default" or internet_access


def _redis(session: Literal["default", "context"] = "default", internet_access: bool = False) -> BaseMemory:
    history_db = RedisChatMessageHistory(session_id=session, url=envs.redis_url)
    return ConversationBufferWindowMemory(
        chat_memory=history_db,
        max_token_limit=1000,
        k=5,
        return_messages=_requires_raw(session, internet_access),
        **_prompt_keys(session, internet_access)
    )


def _redis_with_summary(session: Literal["default", "context"] = "default") -> BaseMemory:
    history_db = RedisChatMessageHistory(session_id=session, url=envs.redis_url)
    memory = ConversationSummaryBufferMemory(
        llm=OpenAI(temperature=0),
        chat_memory=history_db,
        return_messages=_requires_raw(session, False),
        max_token_limit=1000,
        **_prompt_keys(session)
    )
    return memory


RedisSummaryMemory = _redis_with_summary
Redis = _redis
