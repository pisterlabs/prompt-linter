from typing import List, Any, Dict, Optional, Type

from pydantic import BaseModel, Field
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    Callbacks,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from ai.chains.sql.schema_evaluation.column.column_single import (
    ColumneEvaluatedDescription,
    SingleColumnEvaluateDescriptionChain,
    MultipleColumnDescriptionChain,
)


class TableEvaluatedDescription(BaseModel):
    table: str = Field(description="The table name being described")
    possible_description: str = Field(
        description=(
            "A detailed description and discussion of the table. "
            "You are to provide a description the table as a whole. "
            " Responding to queries like:"
            "\n- What kind of data is it?"
            "\n- What is it about?"
            "\n- What is it used for?"
        )
    )
    short_description: str = Field(
        description=(
            "A short summary description of the table and its contents. "
            "This covers all the important details about the table."
        )
    )


class DetailedTableEvaluatedDescription(TableEvaluatedDescription):
    columns: List[ColumneEvaluatedDescription]


EVAL_TABLE_DESC_TEMPLATE = (
    "You will be presented with a a table schema, table extract, and "
    "some guesses about what each column in the table contains. "
    "You are to evaluate what the table contains into a detailed "
    "and short summary. "
    "The descriptions are intended to assist in assessing the table "
    "in later data analysis. "
    "Be factual and specific."
    "\n\nTablename: {table_name}"
    "\n\nTable Info (only a small sample of the data): {table_info}"
    "\n\nInfered Column Descriptions: {infered_columns}"
    "\n\n{format_instructions}"
)


EVAL_TABLE_DESC_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=TableEvaluatedDescription
)

EVAL_TABLE_DESC_PROMPT = PromptTemplate(
    template=EVAL_TABLE_DESC_TEMPLATE,
    input_variables=["table_name", "table_info", "infered_columns"],
    partial_variables={
        "format_instructions": EVAL_TABLE_DESC_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=EVAL_TABLE_DESC_OUTPUT_PARSER,
)


class TableEvaluateDescriptionChain(LLMChain):
    prompt: PromptTemplate = EVAL_TABLE_DESC_PROMPT
    output_parser: PydanticOutputParser = EVAL_TABLE_DESC_PROMPT.output_parser


class TableEvaluationChain(Chain):
    column_description_chain: MultipleColumnDescriptionChain
    table_description_chain: TableEvaluateDescriptionChain
    return_column_descriptions: bool = False
    output_key: str = "table_summary"

    @property
    def input_keys(self) -> List[str]:
        return ["table_name"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, TableEvaluatedDescription | DetailedTableEvaluatedDescription]:
        # enhance inputs onnly once ...
        inputs = self.column_description_chain.input_prepper(inputs)

        column_descriptions = self.column_description_chain.predict(
            **inputs, callbacks=run_manager.inheritable_handlers
        )
        table_description = self.table_description_chain.predict(
            infered_columns=column_descriptions,
            **inputs,
            callbacks=run_manager.inheritable_handlers
        )
        if self.return_column_descriptions:
            return {
                self.output_key: DetailedTableEvaluatedDescription(
                    **table_description.dict(), columns=column_descriptions
                )
            }
        return {self.output_key: table_description}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, TableEvaluatedDescription | DetailedTableEvaluatedDescription]:
        # enhance inputs onnly once ...
        inputs = self.column_description_chain.input_prepper(inputs)

        column_descriptions = await self.column_description_chain.apredict(
            **inputs, callbacks=run_manager.inheritable_handlers
        )
        table_description = await self.table_description_chain.apredict(
            infered_columns=column_descriptions,
            **inputs,
            callbacks=run_manager.inheritable_handlers
        )
        if self.return_column_descriptions:
            return {
                self.output_key: DetailedTableEvaluatedDescription(
                    **table_description.dict(), columns=column_descriptions
                )
            }
        return {self.output_key: table_description}

    def predict(
        self, callbacks: Callbacks = None, **kwargs: Any
    ) -> TableEvaluatedDescription | DetailedTableEvaluatedDescription:
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(
        self, callbacks: Callbacks = None, **kwargs: Any
    ) -> TableEvaluatedDescription | DetailedTableEvaluatedDescription:
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]
