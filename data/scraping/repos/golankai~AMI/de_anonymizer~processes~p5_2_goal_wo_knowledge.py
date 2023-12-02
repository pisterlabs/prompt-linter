from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)


P5_2_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. 
    The AI is helping the user validate their anonymization process, mimicking an intruder.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P5_2_Q1_TEMPLATE = """
    Task: please re-identify the person in the following anonymized text.
    Note that the anonymized text use placeholders. 
    Your goal is to guess the famous person.
    
    {format_instructions}
    
    anonymized text: "{user_input}"
"""


class P5_2_Q1Parser(BaseModel):
    name: str = Field(description="Name of the person or UNKNOWN")
    score: float = Field(
        description="Assign confidence score from 0 to 1 for your name re-identification correctness, where 1 is the most confident"
    )


process_5_2_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P5_2_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P5_2_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P5_2_Q1Parser)),
    ],
)
