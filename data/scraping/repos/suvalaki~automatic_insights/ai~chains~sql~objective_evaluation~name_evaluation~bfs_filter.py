from typing import List, Dict, Optional, Any, Tuple

from pydantic import validator, root_validator

from langchain import PromptTemplate, LLMChain
from langchain.chains.base import Chain
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.manager import (
    Callbacks,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.sql_database import SQLDatabase

# BFS filter does all the tables in paralel
# its probably better to periodically look at the data and just
# write a plain text description of the data.

from ai.chains.sql.objective_evaluation.name_evaluation.base import (
    TableSelectionDetailThought,
)
from ai.chains.parallel_chain import ExtractChain, ParallelChain

ADDITIONAL_CONTEXT = (
    "This is the first part of a process where later you will "
    "query the table schemas for more information in building sql "
    "queries to respond to the objective"
)

SINGLE_EVALUATION_PROMPT_TEMPLATE = (
    "You will be provided with an objective, a table name, and a "
    "list of other tables. "
    "You are to evaluate the relevance of the table (for completing "
    " the objective) from just the name. "
    "Speculate about the table. "
    " Guess what columns might be in the table. "
    "If the table looks useful in combination with another table "
    "then you should say so. "
    " You should be critical when scoring the likelihood."
    "{additional_context}"
    "\n\nObjective: {objective}"
    "\n\nTable: {table}"
    "\n\nTable Schema and information: {table_info}"
    "\n\nOther Tables: {tables}"
    "\n\n{format_instructions}"
)

SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=TableSelectionDetailThought
)

SINGLE_EVALUATION_PROMPT = PromptTemplate(
    template=SINGLE_EVALUATION_PROMPT_TEMPLATE,
    input_variables=["objective", "table", "table_info", "tables"],
    partial_variables={
        "additional_context": ADDITIONAL_CONTEXT,
        "format_instructions": SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER.get_format_instructions(),
    },
    output_parser=SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER,
)


class SingleTablenameRelevanceEvaluationChain(LLMChain):
    prompt: PromptTemplate = SINGLE_EVALUATION_PROMPT
    output_parser: PydanticOutputParser = SINGLE_EVALUATION_PROMPT_OUTPUT_PARSER


def extract_inputs(db, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"table": table, "table_info": db.get_table_info_no_throw([table])}
        for table in inputs["tables"]
    ]


class TableInfoInputsListExtractor(ExtractChain):
    db: SQLDatabase
    input_variables: List[str] = ["tables"]

    @root_validator(pre=True)
    def initialize_transform(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "transform" not in values:
            values["transform"] = lambda x: {
                "extracted": extract_inputs(values["db"], x)
            }
        return values


class MultipleTablenameRelevanceEvaluationChain(ParallelChain):
    llm: BaseLanguageModel
    db: SQLDatabase
    extract_inputs: TableInfoInputsListExtractor
    chain: Chain
    output_key: str = "tablename_evaluations"

    @root_validator(pre=True)
    def initialize_extract_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "extract_inputs" not in values:
            values["extract_inputs"] = TableInfoInputsListExtractor(db=values["db"])
        return values

    @root_validator(pre=True)
    def initialize_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "chain" not in values:
            values["chain"] = SingleTablenameRelevanceEvaluationChain(llm=values["llm"])
        return values

    @property
    def input_keys(self) -> List[str]:
        return ["objective", "tables"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
