from __future__ import annotations
from langchain.chains.base import Chain
from overrides import overrides

from Result import Result


"""Chain for interacting with SQL Database."""


import warnings
from typing import Any, Dict, List, Optional

from langchain import SQLDatabase
from langchain.output_parsers import PydanticOutputParser
from loguru import logger
from pydantic import Extra, Field, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import DECIDER_PROMPT, PROMPT, SQL_PROMPTS
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.tools.sql_database.prompt import QUERY_CHECKER
INTERMEDIATE_STEPS_KEY = "intermediate_steps"
_QUOTRON_TEMPLATE = """Question: What is the highest churn for each year?
            "result": {{ "sql": "SELECT MAX(churn), EXTRACT('year' from date) as year from customer_acquisition_data group by year;", "x_axis": [], "y_axis": ["MAX(churn) "], "time_grain": "P1Y", "chart_type": "echarts_timeseries_bar", "title": "Highest yearly churn"}}
            ###
            Question: What is the highest revenue for each product for each quarter?
            "result": {{ "sql": "SELECT MAX(revenue), product from customer_acquisition_data group by product;", "x_axis": ["product"],"y_axis":[ "MAX(revenue)"],"time_grain": "P3M", "chart_type": "echarts_timeseries_bar", "title": "Highest quarterly revenue by product"}}
            ###
            Question: {query}
            """
QUOTRON_PROMPT = PromptTemplate(
    input_variables=["query"], template=_QUOTRON_TEMPLATE, output_parser=PydanticOutputParser(pydantic_object=Result)
)

class VisualizationChain(Chain):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SQLDatabaseChain.from_llm(OpenAI(), db)
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    database: SQLDatabase = Field(exclude=True)
    """SQL Database to connect to."""
    prompt: Optional[BasePromptTemplate] = None
    """[Deprecated] Prompt to use to translate natural language to SQL."""
    top_k: int = 5
    """Number of results to return from the query"""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""
    use_query_checker: bool = True
    """Whether or not the query checker tool should be used to attempt 
    to fix the initial SQL from the LLM."""
    query_checker_prompt: Optional[BasePromptTemplate] = None
    """The prompt template that should be used by the query checker"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an SQLDatabaseChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                database = values["database"]
                prompt = values.get("prompt") or SQL_PROMPTS.get(
                    database.dialect, PROMPT
                )
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, VisualizationChain.INTERMEDIATE_STEPS_KEY]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        llm_inputs = {
            "query": inputs[self.input_key]
        }
        result = self.llm_chain.predict_and_parse(**llm_inputs)
        return result

    @property
    def _chain_type(self) -> str:
        return "visualization_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        db: SQLDatabase,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> VisualizationChain:
        prompt = QUOTRON_PROMPT
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, database=db, **kwargs)

    @overrides
    def _validate_outputs(self, outputs: Dict[str, Any]) -> None:
        return
