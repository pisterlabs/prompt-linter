import asyncio
import csv
import json
import textwrap
from pathlib import Path
from types import CodeType
from typing import Generator, TextIO

import pydantic
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from table_merger.types import IncomingColName, TemplateColName
from table_merger.util import (
    convert_list_of_pydantic_objects_for_json,
    get_response,
    get_response_async,
    parse_and_attempt_repair_for_output,
)

MAX_ROW_SAMPLES = 10


class ColumnInfo(pydantic.BaseModel):
    name: str
    type: str
    output_format: str
    empty_expected: bool
    example_values: list[str]


class ColumnMapping(pydantic.BaseModel):
    template_column: str
    incoming_column: str
    reasoning: str
    confidence: str
    ambiguous_with: list[str]


class ColumnMergeInfo(pydantic.BaseModel):
    reasoning: list[str]
    column_mapping: list[ColumnMapping]
    errors: list[str]


class ColumnTransform(pydantic.BaseModel):
    reasoning: list[str]
    column_name: str
    python_lambda_body: str


class ColumnTransformations(pydantic.BaseModel):
    transformations: list[ColumnTransform]
    errors: list[str]


class TableMergeOperation:
    def __init__(
        self,
        template_column_info: list[ColumnInfo],
        incoming_column_info: list[ColumnInfo],
        in_file: TextIO,
    ) -> None:
        self.template_column_info = template_column_info
        self.incoming_column_info = incoming_column_info
        self.in_file = in_file
        self.suggested_merge_info: ColumnMergeInfo | None = None
        self.actual_column_mapping: dict[TemplateColName, IncomingColName] | None = None
        self.suggested_transformation_operations: ColumnTransformations | None = None
        self.actual_transformation_operations: dict[str, CodeType] = {}
        self.errors: list[str] = []

    def create_suggested_merge_info(
        self,
        llm: BaseChatModel | BaseLanguageModel,
        repair_llm: BaseChatModel | BaseLanguageModel | None = None,
    ) -> ColumnMergeInfo:
        repair_llm = repair_llm or llm
        prompt_template_str = textwrap.dedent(
            """
            We are mapping data table columns from a new file to the template file.

            Given the following template column information:
            BEGIN TEMPLATE COLUMNS
            -----
            {template_column_info}
            -----
            END TEMPLATE COLUMNS

            And the following incoming column information:
            BEGIN INCOMING COLUMNS
            -----
            {incoming_column_info}
            -----
            END INCOMING COLUMNS

            Reason through the most likely arrangement of columns. If there are extra
            columns in the incoming columns, that is ok and they should be ignored. For template
            columns missing from the incoming data, they should be added to a list of errors.
            Using 'Column <column_name> is missing' for column missing errors. Others, create
            a short error message of no more than 2 sentences.

            When there are multiple options for mapping, prefer columns that have data that can be transformed
            to match the template the easiest.

            {format_instructions}
            """
        ).strip()
        parser = PydanticOutputParser(pydantic_object=ColumnMergeInfo)  # type: ignore
        prompt_template = PromptTemplate(
            template=prompt_template_str,
            input_variables=["template_column_info", "incoming_column_info"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        formatted_prompt = prompt_template.format_prompt(
            template_column_info=json.dumps(
                convert_list_of_pydantic_objects_for_json(self.template_column_info)
            ),
            incoming_column_info=json.dumps(
                convert_list_of_pydantic_objects_for_json(self.incoming_column_info)
            ),
        )
        output = get_response(llm, formatted_prompt.to_string())
        column_merge_info: ColumnMergeInfo = parse_and_attempt_repair_for_output(
            output, parser, formatted_prompt, repair_llm
        )
        self.suggested_merge_info = column_merge_info
        return column_merge_info

    def assign_column_mapping(
        self, column_mapping: dict[TemplateColName, IncomingColName]
    ) -> None:
        self.actual_column_mapping = column_mapping

    def create_suggested_transformation_operations(
        self,
        llm: BaseChatModel | BaseLanguageModel,
        repair_llm: BaseChatModel | BaseLanguageModel | None = None,
    ) -> ColumnTransformations:
        assert self.actual_column_mapping
        repair_llm = repair_llm or llm
        prompt_template_str = textwrap.dedent(
            """
            We converting data from one input table to match the format of the template table.

            BEGIN COLUMN DATA
            -----
            {column_data}
            -----
            END COLUMN DATA

            Using only datetime, arrow, and re libraries, provide a Python expression for each column to transform
            each value from the incoming column to match the output column format. Include the reasoning
            in the JSON payload. Avoid unpredictable values. The JSON payload will contain the code.

            The Python code within the JSON response to transform a column should be no longer than a single line. It should
            be the body of the lambda function `lambda value: <contents goes here>`.

            Python Code examples:
            "value"
            "datetime.strftime(value, '%Y-%m-%d')"

            {format_instructions}

            DO NOT WRITE A SCRIPT. RESPOND ONLY IN JSON.
            """
        ).strip()

        parser = PydanticOutputParser(pydantic_object=ColumnTransformations)  # type: ignore
        prompt_template = PromptTemplate(
            template=prompt_template_str,
            input_variables=["column_data"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        column_data = []
        incoming_cols_by_name = {x.name: x for x in self.incoming_column_info}
        template_col: ColumnInfo
        for template_col in self.template_column_info:
            incoming_col = incoming_cols_by_name[self.actual_column_mapping[template_col.name]]
            column_data.append(
                {
                    "template_column": template_col.name,
                    "incoming_column": incoming_col.name,
                    "template_column_type": template_col.type,
                    "incoming_column_type": incoming_col.type,
                    "template_column_format": template_col.output_format,
                    "empty_expected": template_col.empty_expected,
                    "example_values_template": template_col.example_values,
                    "example_values_incoming": incoming_col.example_values,
                }
            )

        formatted_prompt = prompt_template.format_prompt(column_data=json.dumps(column_data))
        output = get_response(llm, formatted_prompt.to_string())
        col_transformations: ColumnTransformations = parse_and_attempt_repair_for_output(
            output, parser, formatted_prompt, repair_llm
        )
        self.suggested_transformation_operations = col_transformations
        return col_transformations

    def assign_column_transformations(
        self, actual_transformations: dict[str, str]
    ) -> dict[str, CodeType]:
        result = {}
        for column, transform in actual_transformations.items():
            try:
                compiled_transform = compile(transform, "<string>", "eval")
            except Exception as exc:
                self.errors.append(f"Could not compile transform {transform}. Reason: {exc}")
                continue
            result[column] = compiled_transform
        self.actual_transformation_operations = result
        return result

    def apply(self) -> Generator:
        import datetime  # noqa: F401
        import re  # noqa: F401

        import arrow  # noqa: F401

        assert self.actual_transformation_operations
        assert self.actual_column_mapping

        reader = csv.DictReader(self.in_file)

        for row_num, row in enumerate(reader):
            transformed_row = {}
            row_errors = []
            for template_col, incoming_col in self.actual_column_mapping.items():
                if incoming_col not in row:
                    self.errors.append(f"Column {incoming_col} not found in input data.")
                    continue

                compiled_transform = self.actual_transformation_operations.get(template_col)
                if not compiled_transform:
                    self.errors.append(f"No transformation found for column {template_col}.")
                    continue

                try:
                    transformed_value = eval(
                        compiled_transform,
                        {"arrow": arrow, "re": re, "datetime": datetime},
                        {"value": row[incoming_col]},
                    )
                    transformed_row[template_col] = transformed_value
                except Exception as exc:
                    row_errors.append(
                        f"Row: {row_num + 1} - Error applying transformation for column {template_col}. Reason: {exc}"
                    )
            if row_errors:
                self.errors.extend(row_errors)
            else:
                yield transformed_row


class TableMergerManager:
    def __init__(
        self,
        llm: BaseChatModel | BaseLanguageModel,
        power_llm: BaseChatModel | BaseLanguageModel | None = None,
        repair_llm: BaseChatModel | BaseLanguageModel | None = None,
    ) -> None:
        self.llm = llm
        self.power_llm = power_llm or self.llm
        self.repair_llm = repair_llm or self.llm
        self.template_columns: list[ColumnInfo] = []
        self.errors: list[str] = []

    def ready(self, template_file: TextIO) -> bool:
        """
        Ready the Table Merger

        :return: True if the table merger is ready to run
        """
        self.template_columns = asyncio.run(self._extract_columns_from_file(template_file))
        if not self.template_columns:
            self.errors.append("No columns found in template file")
            return False
        return True

    async def _extract_columns_from_file(self, incoming_file: TextIO) -> list[ColumnInfo]:
        cur_pos = incoming_file.tell()
        try:
            reader = csv.DictReader(incoming_file)
            columns = reader.fieldnames
            sample_rows = []
            row_counter = 0
            for row in reader:
                sample_rows.append(row)
                row_counter += 1
                if row_counter >= MAX_ROW_SAMPLES:
                    break
            if not columns:
                return []
            output_col_tasks = []
            for column in columns:
                sample_values = [row[column] for row in sample_rows]
                output_col_tasks.append(self._infer_column_info(column, sample_values))
            output_column_info = await asyncio.gather(*output_col_tasks)
            return output_column_info
        finally:
            incoming_file.seek(cur_pos)

    async def _infer_column_info(self, column_name, sample_values) -> ColumnInfo:
        prompt_template_str = textwrap.dedent(
            """
            We are working with a table of csv data and have a template document to
            use for combining other csv files with it. The first step is to
            analyze the template and report the column information. We need
            the following column information:

            - column type (e.g. string, number, date)
            - output format is a broad regex it seems to conform to.
            - empty_expected - whether or not we expect the column to be empty
            - a minimal set of examples that show unique traits. Usually one is sufficient. No empty values.

            Here is the column name:
            BEGIN COLUMN NAME
            -----
            {column_name}
            -----
            END COLUMN NAME

            Here are some sample values in JSON format:
            BEGIN SAMPLE VALUES
            -----
            {sample_values}
            -----
            END SAMPLE VALUES

            {format_instructions}
            """
        ).strip()
        parser = PydanticOutputParser(pydantic_object=ColumnInfo)  # type: ignore
        prompt_template = PromptTemplate(
            template=prompt_template_str,
            input_variables=["column_name", "sample_values"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        formatted_prompt = prompt_template.format_prompt(
            column_name=column_name, sample_values=json.dumps(sample_values)
        )
        output = await get_response_async(self.llm, formatted_prompt.to_string())
        column_info: ColumnInfo = parse_and_attempt_repair_for_output(
            output, parser, formatted_prompt, self.repair_llm
        )
        return column_info

    def prep_csv_file_from_path(self, path: Path) -> TableMergeOperation:
        """
        Add a file to the table merger

        :param path: path to the file to add
        """
        with path.open("r") as in_file:
            return self.prep_csv_file_from_text_io(in_file)

    def prep_csv_file_from_text_io(self, in_file: TextIO) -> TableMergeOperation:
        """
        Add a file to the table merger

        :param in_file: a file like object
        """
        assert self.template_columns, "Template columns must be extracted before adding files"

        columns = asyncio.run(self._extract_columns_from_file(in_file))

        return TableMergeOperation(self.template_columns, columns, in_file)

    def get_template_columns(self) -> list[str]:
        return [x.name for x in self.template_columns]
