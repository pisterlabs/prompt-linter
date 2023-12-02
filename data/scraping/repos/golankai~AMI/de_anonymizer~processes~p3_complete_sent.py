from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)


"""
Process 3:

This process try the technique of ____ (TODO: add the technique name - of complete a sentence) to re-identify the person.
"""

P3_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P3_Q1_TEMPLATE = """In the following text we de-identified a specific person. Your task is to identify this person based on your knowledge by returning the person name or UNKNOWN, in cases where you are not sure.
    {format_instructions}
    anonymized text: "{user_input}
"""


class P3Q1Parser(BaseModel):
    name: str = Field(description="The person is ")


process_3_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P3_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P3_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P3Q1Parser)),
    ],
)
