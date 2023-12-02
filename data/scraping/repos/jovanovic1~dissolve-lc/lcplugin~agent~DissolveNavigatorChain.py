from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from WebActionIdentifier import WebActionIdentifier
from WebNavigator import WebNavigator


class DissolveNavigatorChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def stringInput(**inputs):
        inputs_str = str(inputs)
        print(inputs_str)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        print("### Dissolve Navigator Chain ###")
        # print('loggg#1')
        #format the prompt with given inputs
        prompt_value = self.prompt.format_prompt(**inputs)
        # print('loggg#2')
        
        #navigator instance
        navigator = WebNavigator()

        #if navigator
        if inputs['url'] != "https://www.logitech.com/en-in":
            #do nothing, no navigation
            newUrl = ""
        else:
            newUrl = navigator._run(prompt_value.text)
        
        print('traverse to url: ',newUrl)

        return {self.output_key: newUrl}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

    @property
    def _chain_type(self) -> str:
        return "dissolve_chain"