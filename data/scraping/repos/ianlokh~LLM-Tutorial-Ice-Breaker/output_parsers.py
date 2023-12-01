from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class PersonIntel(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: List[str] = Field(description="Interesting facts about the person")
    topics_of_interest: List[str] = Field(description="Topics that may interest that interest that person")
    ice_breakers: List[str] = Field(description="Icebreakers to open a conversation with that person")

    def to_dict(self):
        return {"summary": self.summary,
                "facts": self.facts,
                "topics_of_interest": self.topics_of_interest,
                "ice_breakers": self.ice_breakers
                }


person_intel_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=PersonIntel)

# class Summary(BaseModel):
#     summary: str = Field(description="summary")
#     facts: List[str] = Field(description="interesting facts about them")
#
#     def to_dict(self):
#         return {"summary": self.summary, "facts": self.facts}
#
#
# class IceBreaker(BaseModel):
#     ice_breakers: List[str] = Field(description="ice breaker list")
#
#     def to_dict(self):
#         return {"ice_breakers": self.ice_breakers}
#
#
# class TopicOfInterest(BaseModel):
#     topics_of_interest: List[str] = Field(
#         description="topic that might interest the person"
#     )
#
#     def to_dict(self):
#         return {"topics_of_interest": self.topics_of_interest}
#
#
# summary_parser = PydanticOutputParser(pydantic_object=Summary)
# ice_breaker_parser = PydanticOutputParser(pydantic_object=IceBreaker)
# topics_of_interest_parser = PydanticOutputParser(pydantic_object=IceBreaker)
