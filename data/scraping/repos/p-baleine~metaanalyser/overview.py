from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts.base import BasePromptTemplate
from typing import Any, Dict, List, Optional

from ...paper import (
    Paper,
    get_abstract_with_token_limit,
    get_categories_string,
)
from ..base import (
    SRBaseChain,
    maybe_retry_with_error_output_parser,
)
from .prompt import OVERVIEW_PROMPT, output_parser


class SROverviewChain(SRBaseChain):

    prompt: BasePromptTemplate = OVERVIEW_PROMPT
    nb_categories: int = 3
    nb_token_limit: int = 1_500
    nb_max_retry: int = 3

    @property
    def input_keys(self) -> List[str]:
        return ["query", "papers"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        input_list = get_input_list(
            self.llm,
            inputs["query"],
            inputs["papers"],
            self.nb_categories,
            self.nb_token_limit,
        )
        output = super()._call(input_list, run_manager=run_manager)
        return maybe_retry_with_error_output_parser(
                llm=self.llm,
                input_list=input_list,
                output=output,
                output_parser=output_parser,
                output_key=self.output_key,
                prompt=self.prompt,
        )

    def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        input_list = get_input_list(
            self.llm,
            inputs["query"],
            inputs["papers"],
            self.nb_categories,
            self.nb_token_limit,
        )
        output = super()._acall(input_list, run_manager=run_manager)
        return maybe_retry_with_error_output_parser(
                llm=self.llm,
                input_list=input_list,
                output=output,
                output_parser=output_parser,
                output_key=self.output_key,
                prompt=self.prompt,
        )


def get_input_list(
        llm: BaseLanguageModel,
        query: str,
        papers: List[Paper],
        nb_categories: int,
        nb_token_limit: int,
):
    return [{
        "query": query,
        "categories": get_categories_string(papers, nb_categories),
        "abstracts": get_abstract_with_token_limit(llm, papers, nb_token_limit)
    }]
