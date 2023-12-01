import json
import os
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import RegexParser

from processor import DataFrameProcessor
from utils.logging import get_basic_stdout_logger
from utils.templates import matching_template, conclusion_template, MATCHING_RESULT_NAME, MATCHING_RESULT_REGEX, \
    validation_template


def format_dataframe(df: pd.DataFrame, value_separator=',', line_separator='\n', n_rows=None) -> str:
    output = []

    # Use all rows if n_rows is None, otherwise use the specified number of rows
    n = len(df) if n_rows is None else min(n_rows, len(df))

    for col in df.columns:
        line = str(col) + ':'
        line = line + value_separator.join([str(v) for v in df[col].values[:n]])
        output.append(line)

    return line_separator.join(output)


def generate_transformation_functions(llm: BaseChatModel, template_table: pd.DataFrame, source_table: pd.DataFrame,
                                      n_rows: int = 5) -> Dict[str, str]:
    """
    Takes template table and input table and returns a dictionary of transformation functions.
    Format of output: {'transformation_name': 'transformation_function'}
    :param llm: BaseChatModel - Any Langchain LLM wrapper
    :param template_table: pd.DataFrame
    :param source_table: pd.DataFrame
    :param n_rows: int // Number of rows to use as for the value example. Choose as many as you think will represent
    the data disperse well. Usually 5-10 rows are enough.
    :return: Dict[str, str]
    """
    assert template_table.shape[0] > 0, "template_table must have at least one row"
    assert source_table.shape[0] > 0, "source_table must have at least one row"
    str_template_table = format_dataframe(template_table, n_rows=n_rows)
    str_source_table = format_dataframe(source_table, n_rows=n_rows)

    matching_output_parser = RegexParser(regex=MATCHING_RESULT_REGEX, output_keys=[MATCHING_RESULT_NAME])
    matching_prompt = PromptTemplate(input_variables=["template_table", "source_table"], template=matching_template,
                                     output_parser=matching_output_parser)
    validation_prompt = PromptTemplate(input_variables=[MATCHING_RESULT_NAME, "template_table", "source_table"],
                                       template=validation_template, output_parser=matching_output_parser)
    conclusions_prompt = PromptTemplate(input_variables=[MATCHING_RESULT_NAME], template=conclusion_template)

    # Original plan was to use SequentialChain or Agent with tools, but somewhy output parsing doesn't work.
    # So I decided to save time and do it the way it's done now.
    thoughts_chain = LLMChain(llm=llm, prompt=matching_prompt, output_key=MATCHING_RESULT_NAME, verbose=True)
    validation_chain = LLMChain(llm=llm, prompt=validation_prompt, output_key=MATCHING_RESULT_NAME, verbose=True)
    conclusions_chain = LLMChain(llm=llm, prompt=conclusions_prompt, verbose=True)

    thoughts = thoughts_chain.run(template_table=str_template_table, source_table=str_source_table)
    thoughts = matching_output_parser.parse(thoughts)
    validation_kwargs = {MATCHING_RESULT_NAME: thoughts[MATCHING_RESULT_NAME], "template_table": str_template_table,
                         "source_table": str_source_table}
    validation = validation_chain.run(**validation_kwargs)
    validation = matching_output_parser.parse(validation)

    raw_response = json.loads(conclusions_chain.run(validation[MATCHING_RESULT_NAME]))
    # TODO: If output format will be changed, this will need to be changed as well (
    response = {k: v[2] for k, v in raw_response.items()}
    return response, raw_response


if __name__ == "__main__":

    logger = get_basic_stdout_logger()

    if not os.getenv('OPENAI_API_KEY'):
        logger.info("OPENAI_API_KEY not found in environment variables. Loading from .env file")
        load_dotenv()

    template_table = pd.read_table('template.csv', sep=',')
    source_table = pd.read_table('table_A.csv', sep=',')

    # If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
    llm = ChatOpenAI(model=os.getenv('OPENAI_MODEL_NAME'), temperature=os.getenv('OPENAI_TEMPERATURE'))
    transformations, raw = generate_transformation_functions(llm=llm, template_table=template_table,
                                                             source_table=source_table, n_rows=5)
    processor = DataFrameProcessor.from_dict(transformations=transformations)

    output = processor(source_table)
    output.to_csv('output.csv', index=False)
