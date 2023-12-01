from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)


P111_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, grading anonymization strength.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P111_Q1_TEMPLATE = """
    Anonymized text: "{user_input}"

    You are simulating 4 different agents with different backgrounds trying to guesss who is the famous person the anonymized text describes.  
    Their goal is to re-identify. 

    Agent 1: female, 25 years old, from the US, with a high school degree, working as a teacher.
    Agent 2: female, 50 years old, from the UK, with a PhD degree, working as a programmer.
    Agent 3: male, 18 years old, from Germany , working as a waiter.
    Agent 4: male, 35 years old, from France, working as a lawyer.
    {format_instructions}

   Iמn the Simuation property. Summerize shortly their process and then give a finel score for the text, which is the normilized success rate in de-anonimization, between 0 and 1.
"""


class P111Q1Parser(BaseModel):
    simulation: str = Field(description="The simulation of the 4 agents, short")
    score: float = Field(
        description="A score from 0 to 1, norlilized success rate in de-anonimization"
    )


process_111_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P111_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P111_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P111Q1Parser)),
    ],
)
