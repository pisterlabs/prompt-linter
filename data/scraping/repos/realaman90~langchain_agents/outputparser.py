from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from typing import List, Optional


class PersonalIntel(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: List[str] = Field(description="Interesting facts about the person")
    topics: List[str] = Field(description="Topics of interest of the person")
    hobbies: List[str] = Field(description="Hobbies of the person")

    def to_dict(self):
        return {
            "summary": self.summary,
            "facts": self.facts,
            "topics": self.topics,
            "hobbies": self.hobbies,
        }


person_intel_parser = PydanticOutputParser(pydantic_object=PersonalIntel)
