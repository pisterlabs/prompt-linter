"""LLM Chain specifically for evaluating question answering."""
from __future__ import annotations

from typing import Any, List

from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain

from demo.chains.oncall_agent.oncall_agent_prompt import PROMPT

class OncallChain(LLMChain):
    """LLM Chain specifically for evaluating SDR task performance."""

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: PromptTemplate = PROMPT, **kwargs: Any
    ) -> OncallChain:
        """Load SDR Eval Chain from LLM.

        Args:
            llm (BaseLanguageModel): the base language model to use.

            prompt (PromptTemplate): A prompt template containing the input_variables:
            'input', 'answer' and 'result' that will be used as the prompt
            for evaluation.
            Defaults to PROMPT.

            **kwargs: additional keyword arguments.

        Returns:
            QAEvalChain: the loaded QA eval chain.
        """
        expected_input_vars = {"runbook", "error"}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )
        return cls(llm=llm, prompt=prompt, **kwargs)
