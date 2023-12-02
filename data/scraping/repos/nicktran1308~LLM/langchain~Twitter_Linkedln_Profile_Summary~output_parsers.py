# ------------------------------- Import Libraries -------------------------------
from typing import List
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ------------------------------- Summary Model -------------------------------
class Summary(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")

    def to_dict(self) -> dict:
        """Converts the model to dictionary."""
        return {"summary": self.summary, "facts": self.facts}

# ------------------------------- Ice Breaker Model -------------------------------
class IceBreaker(BaseModel):
    ice_breakers: List[str] = Field(description="ice breaker list")

    def to_dict(self) -> dict:
        """Converts the model to dictionary."""
        return {"ice_breakers": self.ice_breakers}

# ------------------------------- Topic of Interest Model -------------------------------
class TopicOfInterest(BaseModel):
    topics_of_interest: List[str] = Field(
        description="topics that might interest the person"
    )

    def to_dict(self) -> dict:
        """Converts the model to dictionary."""
        return {"topics_of_interest": self.topics_of_interest}

# ------------------------------- Parsers Initialization -------------------------------
summary_parser = PydanticOutputParser(pydantic_object=Summary)
ice_breaker_parser = PydanticOutputParser(pydantic_object=IceBreaker)
topics_of_interest_parser = PydanticOutputParser(pydantic_object=TopicOfInterest)

