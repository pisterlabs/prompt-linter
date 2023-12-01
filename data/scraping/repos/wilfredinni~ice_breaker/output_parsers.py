from typing import List

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class PersonIntel(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: List[str] = Field(description="Interesting facts about the person")
    topics_of_interest: List[str] = Field(
        description="Topics that may interest the person"
    )
    ice_breaker: List[str] = Field(
        description="Create Ice Breaker to open a conversation"
    )

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "facts": self.facts,
            "topics_of_interest": self.topics_of_interest,
            "ice_breakers": self.ice_breaker,
        }


person_intel_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=PersonIntel
)
