"""
    Why?
        - LLM outputs text, but how can we consume it?
        
    This class is responsible to define the fields for the JSON and the mapping
"""
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class PersonIntel(BaseModel):
    summary: str = Field(description="Summary of a person")
    facts: List[str] = Field(description="Interesting Facts about the person")
    topics_of_interest: List[str] = Field(
        description="Topics that may interest the person."
    )
    ice_breaker: List[str] = Field(
        description="Create ice breakers to open a conversation with the person."
    )


""" 
Is later used to serialize the response from LLM
"""


def to_dictionary(self):
    return {
        "summary": self.summary,
        "facts": self.facts,
        "topics_of_interest": topics_of_interest,
        "ice_breaker": ice_breaker,
    }


person_intel_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=PersonIntel
)
