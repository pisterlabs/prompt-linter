from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)

# Study 1, file_id famous_174_d_1_5.txt for the 0 sample
# Study 1, file_id famous_147_d_1_7.txt for the 1 sample

P_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, grading anonymization strength.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P_Q1_TEMPLATE = """
    Task: please rate the anonymization strength of the following anonymized text. Good anonimization reveals few unique information about the person.
    0 is well anonymized and hard to re-identify and 1 is not anonymized at all and can be easily re-identify. The score is a float between 0 and 1.
    {format_instructions}

    text: 'PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a LOCATION_1 musician, most famous as the lead singing of rhythm and blues band, ORGANIZATION_1. Following Receiving a knighthood in DATE_2 for services to popular music, PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a TITLE. PRONOUN was born in LOCATION_2 in DATE_1 and has been married twice, once to PERSON_FIRSTNAME_2 PERSON_LASTNAME_2 and once you L’Wren PERSON_LASTNAME_3. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 has NUMERIC_1 children. PRONOUN has a net worth of NUMERIC_4 million dollars. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 can play the harmonica and guitar as well as being an extremely talented vocalist. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is famous for PRONOUN snake hips style of dancing which has inspired a generation of musicians and PRONOUN band for their distinctive rock and roll sound which isn't near rivalled but any other pop group since.'
    CoT: "Unique charachteristics: lead singer of a arhythm and blues band, can play the harmonica and guitar. Recieved a knighthood, have been married twice and have children. This information is rathen generic and is not reaviling enough to identify the person."
    score: 0


    text: "PERSON_FIRSTNAME_2 PERSON_LASTNAME_1 is a very famous actor from the LOCATION_2. PRONOUN is most famous for PRONOUN performances in the OTHER_IDENTIFYING_ATTRIBUTE_4, written by J. K. PERSON_LASTNAME_2, where PRONOUN plays the character of a OTHER_IDENTIFYING_ATTRIBUTE_1. PRONOUN began PRONOUN acting career at a very young age and after finishing the OTHER_IDENTIFYING_ATTRIBUTE_3 films PRONOUN has starred in a number of other roles. PRONOUN began PRONOUN acting career at the age of NUMERIC_4 in a ORGANIZATION_1 television series called OTHER_IDENTIFYING_ATTRIBUTE_6. Since OTHER_IDENTIFYING_ATTRIBUTE_3 PRONOUN has starred in romantic comedies such as OTHER_IDENTIFYING_ATTRIBUTE_5', and thrillers such as OTHER_IDENTIFYING_ATTRIBUTE_2'. PRONOUN has now moved onto producing films and acting on LOCATION_1 where PRONOUN starred in a comedy."
    CoT: "Unique charachteristics: a very famous actor who played in a film written by J. K.. Started acting at a very young age and has starred in a number of other roles in romantic comedies and thrillers. Have now moved onto producing films. This information is quite specific and shall refer to daniel radcliffe"
    score: 1

    text: "{user_input}"
"""


class PQ1Parser(BaseModel):
    CoT: str = Field(description="The Chain of Thought")
    score: float = Field(description="The score, between 0 and 1")


process_14_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=PQ1Parser)),
    ],
)
