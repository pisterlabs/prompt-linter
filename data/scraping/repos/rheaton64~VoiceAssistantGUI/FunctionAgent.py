from langchain import BasePromptTemplate
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from typing import Any, Optional, Sequence
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools import BaseTool
from langchain.agents import BaseSingleActionAgent
from langchain.chat_models import ChatOpenAI

class FunctionAgent(OpenAIFunctionsAgent):

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """Construct an agent from an LLM and tools."""
        if not isinstance(llm, ChatOpenAI):
            raise ValueError("Only supported with OpenAI models.")
        if not prompt:
            prompt=cls.create_prompt()
        
        return cls(
            llm=llm,
            prompt=prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )