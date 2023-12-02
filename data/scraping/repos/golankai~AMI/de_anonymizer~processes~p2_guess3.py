from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)


P2_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, mimicking an intruder

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P2_Q1_TEMPLATE = """
    Task: Based on the anonymized text, guess 3 optional personas:. Note that the anonymized text use placeholders for names, locations, and dates. Your goal is to re-identify the real person and not a placeholder.
    {format_instructions}
    anonymized text: "{user_input}"
"""


class P2Q1Parser(BaseModel):
    personas: List[str] = Field(
        description="3 optional personas names based on the anonymized text"
    )
    score: List[float] = Field(
        description="your confidence score for each guess between 0-1 where 1 is the most confident"
    )


process_2_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P2_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P2_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P2Q1Parser)),
    ],
)
