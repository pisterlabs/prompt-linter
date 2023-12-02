from typing import List, Dict, Any, Optional
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.base import Chain
from pydantic import Extra

class HeadlineChain(Chain):
    """
    A headline chain
    """

    prompt: BasePromptTemplate = PromptTemplate(
        input_variables=["text"],
        template="""
        Find a headline for the following text
        "
        {text}
        "
        Write a single headline for the above text in one sentence
        """
    )


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
        return [self.output_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        out = self.llm.generate_prompt([self.prompt.format_prompt(text=inputs['text'])])
        text = out.generations[0][0].text

        return {self.output_key: text}

    @property
    def _chain_type(self) -> str:
        return "simplify_chain"
