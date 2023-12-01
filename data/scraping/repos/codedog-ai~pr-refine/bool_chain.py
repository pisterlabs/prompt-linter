from typing import Any, Dict, List, Optional

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.pydantic_v1 import Extra
from langchain.schema.language_model import BaseLanguageModel

bool_template = """Answer given question: {question}.

Return True or False only, without other words or comma or symbol.
For example, you can return true or false.

return:"""


class BoolChain(Chain):
    """A llm chain always return True/False"""

    @property
    def lc_serializable(self) -> bool:
        return True

    question_chain: LLMChain
    output_key: str = "flag"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def _chain_type(self) -> str:
        return "bool_chain"

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return self.question_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, question: str):
        """Load chain from llm."""
        question_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(
                bool_template.format(question=question)
            ),
        )
        return cls(question_chain=question_chain)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # type: ignore[call-arg]
        answer = self.question_chain.run(callbacks=_run_manager.get_child(), **inputs)

        flag = self._parse_flag(answer)
        result: Dict[str, Any] = {self.output_key: flag}

        return result

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()

        answer = await self.question_chain.arun(
            callbacks=_run_manager.get_child(), **inputs
        )

        flag = self._parse_flag(answer)
        result: Dict[str, Any] = {
            self.output_key: flag,
        }
        return result

    def _parse_flag(self, answer: str) -> bool:
        if answer[:4].lower() == "true":
            return True
        elif answer[:5].lower() == "false":
            return False

        raise ValueError(f"Cannot parse answer {answer} to bool")


if __name__ == "__main__":
    from pr_refine.utils import load_gpt35_llm

    chain = BoolChain.from_llm(load_gpt35_llm(), "Does one plus one equal three ?")
    print(chain.run(a="1"))
