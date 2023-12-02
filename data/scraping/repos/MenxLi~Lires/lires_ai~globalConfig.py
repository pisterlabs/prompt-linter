"""
Usage: 
import globalConfig as config
...
"""
from __future__ import annotations
import openai
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .lmInterface import ChatStreamIterType

# may be changed by command line arguments
local_llm_name: Optional[ChatStreamIterType] = None
local_llm_load_bit: Literal[16, 8, 4] = 8

# this specifies the default chat api for the models,
# other models will fallback to using basaran's huggingface api as local model
# may be changed by command line arguments
openai_api_chat_models = [
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", 
]

def defaultChatModel() -> ChatStreamIterType:
    if local_llm_name is not None:
        return local_llm_name
    else:
        return openai_api_chat_models[0]