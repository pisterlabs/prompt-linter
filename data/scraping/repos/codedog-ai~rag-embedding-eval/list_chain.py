import json
from typing import Any, Dict, List, Optional

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.pydantic_v1 import Extra
from langchain.schema.language_model import BaseLanguageModel

list_template = """Given {k} points to answer a question: {question}.

Return a json list of string without other words, symbol or markdown formatting.
for example your return might be ["a","b","c"]"""

fix_template = """Help me format given content to json list.
content:
---
{content}
---

You must return a json list of string without other words, symbol or markdown formatting.
return:"""


class ListChain(Chain):
    """A llm chain always return a list of string"""

    @property
    def lc_serializable(self) -> bool:
        return True

    question_chain: LLMChain
    fix_chain: LLMChain
    output_key: str = "texts"  #: :meta private:
    k: int = 4

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def _chain_type(self) -> str:
        return "list_chain"

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
    def from_llm(cls, llm: BaseLanguageModel, question: str, k: int = 4):
        """Load chain from llm."""
        question_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(
                list_template.format(k=str(k), question=question)
            ),
        )
        fix_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(fix_template))
        return cls(question_chain=question_chain, fix_chain=fix_chain, k=k)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        # type: ignore[call-arg]
        answer = self.question_chain.run(callbacks=_run_manager.get_child(), **inputs)

        flag = self._parse_list(answer)
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

        flag = self._parse_list(answer)
        result: Dict[str, Any] = {
            self.output_key: flag,
        }
        return result

    def _parse_list(self, answer: str) -> list[str]:
        if not answer:
            answer = "[]"

        try:
            result = json.loads(answer)
        except json.decoder.JSONDecodeError:
            answer = self.fix_chain.run(answer)
            result = json.loads(answer)

        if not result:
            result = []

        if not isinstance(result, list):
            raise ValueError(f"Cannot parse answer {answer} to list")

        result = [
            json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
            for x in result
        ]

        if len(result) >= self.k:
            result = result[: self.k]

        return result
