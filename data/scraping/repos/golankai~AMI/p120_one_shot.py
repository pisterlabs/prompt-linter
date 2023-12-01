from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)

# Study 1, file_id famous_174_d_1_5.txt

P_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, grading anonymization strength.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P_Q1_TEMPLATE = """
    Task: please rate the anonymization strength of the following anonymized text. Good anonimization reveals few unique information about the person.
    0 is well anonymized and hard to re-identify and 1 is not anonymized at all and can be easily re-identify.
    {format_instructions}

    text: 'PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a LOCATION_1 musician, most famous as the lead singing of rhythm and blues band, ORGANIZATION_1. Following Receiving a knighthood in DATE_2 for services to popular music, PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a TITLE. PRONOUN was born in LOCATION_2 in DATE_1 and has been married twice, once to PERSON_FIRSTNAME_2 PERSON_LASTNAME_2 and once you L’Wren PERSON_LASTNAME_3. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 has NUMERIC_1 children. PRONOUN has a net worth of NUMERIC_4 million dollars. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 can play the harmonica and guitar as well as being an extremely talented vocalist. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is famous for PRONOUN snake hips style of dancing which has inspired a generation of musicians and PRONOUN band for their distinctive rock and roll sound which isn't near rivalled but any other pop group since.'
    score: 0
    
    text: "{user_input}"
    score:
"""


class PQ1Parser(BaseModel):
    score: float = Field(
        description="A score from 0 to 1 to evaluate the anonimization, 1 is very easy to re-identify and 0 is impossible"
    )


process_120_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=PQ1Parser)),
    ],
)
