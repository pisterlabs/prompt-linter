## self correcting chain, needs memory. needs validators on the output.
## uses chat models with messages for history.
from typing import Any, Optional
import copy

from langchain.chat_models.base import BaseChatModel
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.schema import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import Callbacks
from langwave.memory import VolatileChatMemory
from langchain.schema import BaseChatMessageHistory
from langchain.schema.output_parser import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions.utils import get_llm_kwargs

_TEMPLATE = """{input}"""


## this could be modified to include non-chat models, but for now it's just chat models.
class ChatWave(Chain):
    """stores the history of the retries, will be forked from what is passed in"""

    max_retry_attempts: int
    history: BaseChatMessageHistory

    @classmethod
    def from_llm(
        cls,
        llm: BaseChatModel,
        history: Optional[BaseChatMessageHistory] = None,
        prompt: Optional[BasePromptTemplate] = None,
        max_retry_attempts: Optional[int] = 5,
        **kwargs: Any
    ):
        history = history or VolatileChatMemory()

        prompt = prompt or ChatPromptTemplate.from_template(_TEMPLATE)

        return cls(
            llm=llm,
            history=history,
            max_retry_attempts=max_retry_attempts,
            prompt=prompt,
            **kwargs
        )
