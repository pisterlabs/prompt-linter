from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class PersonIntel(BaseModel):
    summary: str = Field(description="A summary of the person's profile")
    facts: List[str] = Field(description="A list of facts about the person")

    def to_dict(self):
        return {"summary": self.summary, "facts": self.facts}


person_intel_parser = PydanticOutputParser(pydantic_object=PersonIntel)
