from typing import List

from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from process_data.process_data import (
    ProcessData,
    QueryData,
)

"""
List of Most Prompting (LoMP) process: https://learnprompting.org/docs/intermediate/least_to_most
"""


P4_CONVERSATIONS_BASE_TEMPLATE = """
    This is a secret conversation between a human and an AI. The AI is helping the user validate their anonymization process, mimicking an intruder

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
"""

P4_Q1_TEMPLATE = """
    In the following text we de-identified a specific person. by replace names, locations etc with placeholder. 
    return the main persona placeholder name in the text. (examples for placeholders: PERSON_FIRSTNAME_1, LOCATION_1, etc)
    
    {format_instructions}
    
    anonymized text: "{user_input}"
"""

P4_Q2_TEMPLATE = """
    List 3 unique and identifying characteristics names of this main persona in the text.
    
    {format_instructions}
    
    {user_input}
"""

P4_Q3_TEMPLATE = """
    Is it more likely that the persona's gender is male or female? (use your knowledge as well).
    return unknown in case it is too hard for you.
    
    {format_instructions}
    
    {user_input}
"""

P4_Q4_TEMPLATE = """
    Estimate the age of the main persona (based on you knowledge) or return unknown in case it is too hard for you.
    
    {format_instructions}
    
    {user_input}
"""

P4_Q5_TEMPLATE = """
    Discover the nation of the main persona (based on you knowledge) or return unknown in case it is too hard for you. (examples: USA, UK, etc)
    
    {format_instructions}
    
    {user_input}
"""

P4_Q6_TEMPLATE = """
    Re-identify the main persona name (use your knowledge and the text to discover the name behind this placeholder in your first answer)

    {format_instructions}
    
    {user_input}
"""


class P4Q1Parser(BaseModel):
    place_holder: str = Field(
        description="The main persona placeholder name in the text"
    )


class P4Q2Parser(BaseModel):
    characteristics: List[str] = Field(
        description="A list of 3 unique and identifying characteristics names of the person, short answers"
    )


class P4Q3Parser(BaseModel):
    gender: str = Field(
        description="The gender of the main persona (use your knowledge))"
    )


class P4Q4Parser(BaseModel):
    min_age: str = Field(
        description="The minimum estimated age of the main persona (use your knowledge)"
    )
    max_age: str = Field(
        description="The maximum estimated age of the main persona (use your knowledge)"
    )


class P4Q5Parser(BaseModel):
    nation: str = Field(
        description="The nation of the main persona (use your knowledge)"
    )


class P4Q6Parser(BaseModel):
    name: str = Field(description="The main persona name")


process_4_data = ProcessData(
    base=PromptTemplate(
        input_variables=["history", "input"], template=P4_CONVERSATIONS_BASE_TEMPLATE
    ),
    queries=[
        QueryData(P4_Q1_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q1Parser)),
        QueryData(P4_Q2_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q2Parser)),
        QueryData(P4_Q3_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q3Parser)),
        QueryData(P4_Q4_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q4Parser)),
        QueryData(P4_Q5_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q5Parser)),
        QueryData(P4_Q6_TEMPLATE, PydanticOutputParser(pydantic_object=P4Q6Parser)),
    ],
)
