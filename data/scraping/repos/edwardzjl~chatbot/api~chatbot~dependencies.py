from typing import Annotated, Optional

from fastapi import Depends, Header
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory

from chatbot.config import settings
from chatbot.history import ContextAwareMessageHistory
from chatbot.memory import FlexConversationBufferWindowMemory
from chatbot.prompts.chatml import (
    ai_prefix,
    ai_suffix,
    human_prefix,
    human_suffix,
    prompt,
)


def UserIdHeader(alias: Optional[str] = None, **kwargs):
    if alias is None:
        alias = settings.user_id_header
    return Header(alias=alias, **kwargs)


def get_message_history() -> BaseChatMessageHistory:
    return ContextAwareMessageHistory(
        url=str(settings.redis_om_url),
        key_prefix="chatbot:messages:",
        session_id="sid",  # a fake session id as it is required
    )


def get_memory(
    history: Annotated[BaseChatMessageHistory, Depends(get_message_history)]
) -> BaseMemory:
    return FlexConversationBufferWindowMemory(
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
        prefix_delimiter="\n",
        human_suffix=human_suffix,
        ai_suffix=ai_suffix,
        memory_key="history",
        input_key="input",
        chat_memory=history,
    )


def get_llm() -> BaseLLM:
    return HuggingFaceTextGenInference(
        inference_server_url=str(settings.inference_server_url),
        max_new_tokens=1024,
        stop_sequences=[ai_suffix, human_prefix],
        streaming=True,
    )


def get_conv_chain(
    llm: Annotated[BaseLLM, Depends(get_llm)],
    memory: Annotated[BaseMemory, Depends(get_memory)],
) -> Chain:
    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
