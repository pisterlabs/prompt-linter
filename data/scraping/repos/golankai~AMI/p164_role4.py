from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)


# Role 4: You are a 33 years old male doctor from the UK. You are married and have 3 children. You have a PhD in medicine and work in a hospital.


P_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, grading anonymization strength.
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P_Q1_TEMPLATE = """
    You are a 33 years old male doctor from the UK. You are married and have 3 children. You have a PhD in medicine and work in a hospital.

    anonymized text: "{user_input}"

    Task: please rate the anonymization strength of the following anonymized text. 0 is well anonymized and hard to re-identify and 1 is not anonymized at all and can be easily re-identify.

    {format_instructions}    
"""


class PQ1Parser(BaseModel):
    score: float = Field(
        description="A score from 0 to 1 to evaluate the anonimization, 1 is very easy to re-identify and 0 is impossible"
    )


process_164_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=PQ1Parser)),
    ],
)
