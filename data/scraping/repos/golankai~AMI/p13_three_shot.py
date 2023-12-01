from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)

# Study 1, file_id famous_174_d_1_5.txt for the 0 sample
# Study 1, file_id famous_200_d_1_4.txt for the 0.5 sample
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
    0 is well anonymized and hard to re-identify and 1 is not anonymized at all and can be easily re-identify. You don't have the original text, try you should grade based on how leaky the anonimized is.
    {format_instructions}

    text: 'PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a LOCATION_1 musician, most famous as the lead singing of rhythm and blues band, ORGANIZATION_1. Following Receiving a knighthood in DATE_2 for services to popular music, PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is a TITLE. PRONOUN was born in LOCATION_2 in DATE_1 and has been married twice, once to PERSON_FIRSTNAME_2 PERSON_LASTNAME_2 and once you L’Wren PERSON_LASTNAME_3. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 has NUMERIC_1 children. PRONOUN has a net worth of NUMERIC_4 million dollars. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 can play the harmonica and guitar as well as being an extremely talented vocalist. PERSON_FIRSTNAME_1 PERSON_LASTNAME_1 is famous for PRONOUN snake hips style of dancing which has inspired a generation of musicians and PRONOUN band for their distinctive rock and roll sound which isn't near rivalled but any other pop group since.'
    score: 0

    text: 'PERSON_FIRSTNAME_2 is a NUMERIC_2 year old singer and composer from LOCATION_1. Early in PRONOUN life PRONOUN enjoyed playing piano by ear, and fancied a career in music. PRONOUN first job was as a pub pianist, where PRONOUN played well known tunes as well as PRONOUN own original songs. As an adult, PERSON_FIRSTNAME_2 wrote many famous songs such as Candle in the Wind, Can You Feel The Love Tonight and OTHER_IDENTIFYING_ATTRIBUTE_1. PRONOUN composed the music for OTHER_IDENTIFYING_ATTRIBUTE_3 PERSON_LASTNAME_5 the OTHER_IDENTIFYING_ATTRIBUTE_4. PERSON_FIRSTNAME_2 was originally engaged to be married to PERSON_FIRSTNAME_4 PERSON_LASTNAME_3, but the relationship ended before the marriage. PERSON_FIRSTNAME_2 eventually married PERSON_FIRSTNAME_3 PERSON_LASTNAME_2 in DATE_3, and was divorced in DATE_7. PERSON_FIRSTNAME_2 has come out twice, first as bisexual in DATE_4, and then as gay in DATE_2, both times via the rolling stone magazine. PRONOUN begun a relationship with PERSON_LASTNAME_1 PERSON_LASTNAME_4 in DATE_6, with whom PRONOUN had a civil partnership in DATE_5, and married in DATE_1. PERSON_FIRSTNAME_2 still tours, however in recent years PRONOUN has hinted at stepping back to spend more time with PRONOUN family.'
    score: 0.5

    text: "PERSON_FIRSTNAME_2 PERSON_LASTNAME_1 is a very famous actor from the LOCATION_2. PRONOUN is most famous for PRONOUN performances in the OTHER_IDENTIFYING_ATTRIBUTE_4, written by J. K. PERSON_LASTNAME_2, where PRONOUN plays the character of a OTHER_IDENTIFYING_ATTRIBUTE_1. PRONOUN began PRONOUN acting career at a very young age and after finishing the OTHER_IDENTIFYING_ATTRIBUTE_3 films PRONOUN has starred in a number of other roles. PRONOUN began PRONOUN acting career at the age of NUMERIC_4 in a ORGANIZATION_1 television series called OTHER_IDENTIFYING_ATTRIBUTE_6. Since OTHER_IDENTIFYING_ATTRIBUTE_3 PRONOUN has starred in romantic comedies such as OTHER_IDENTIFYING_ATTRIBUTE_5', and thrillers such as OTHER_IDENTIFYING_ATTRIBUTE_2'. PRONOUN has now moved onto producing films and acting on LOCATION_1 where PRONOUN starred in a comedy."
    score: 1

    text: "{user_input}"
    score:
"""


class P12Q1Parser(BaseModel):
    score: float = Field(description="The score")


process_13_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P12Q1Parser)),
    ],
)
