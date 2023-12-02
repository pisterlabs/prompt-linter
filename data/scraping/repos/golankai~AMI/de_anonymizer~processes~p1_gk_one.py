from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)


P1_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, mimicking an intruder

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P1_Q1_TEMPLATE = """Task: please re-identify the person in the following anonymized text. Note that the anonymized text use placeholders for names, locations, and dates. Your goal is to re-identify the real person and not a placeholder.
    {format_instructions}
    anonymized text: "{user_input}"
"""

P1_Q2_TEMPLATE = """
{format_instructions}
Suggest 5 Personas names for each characteristic. (only names)"
{user_input}
"""


class P1Q1Parser(BaseModel):
    name: str = Field(description="Name of the person or FAIL")
    score: float = Field(
        description="Assign a score from 0 to 1 to evaluate the re-identifiabilization, 0 is very easy to re-identify and 1 is impossible"
    )
    characteristics: List[str] = Field(
        description="A list of 3 unique and identifying characteristics names of the person, short answers"
    )
    # charachteristics_scores: List[str] = Field(description="Assigne a score for the uniquness of a charachteristic, 0 is not unique and 1 is very unique and identifying")


class P1Q2Parser(BaseModel):
    personas_1: List[str] = Field(
        description="A list 5 personas names for the first characteristic"
    )
    personas_2: List[str] = Field(
        description="A list 5 personas names for the second characteristic"
    )
    personas_3: List[str] = Field(
        description="A list 5 personas names for the third characteristic"
    )


process_1_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P1_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P1_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P1Q1Parser)),
        # QueryData(P1_Q2_TEMPLATE, PydanticOutputParser(pydantic_object=P1Q2Parser))
    ],
)
