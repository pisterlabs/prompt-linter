from typing import List

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    summary: str = Field(
        description="A one paragraph reflection of the conversation, talking about how you feel about how compatible a friendship between you and the other person would be."
    )
    reasons_for: List[str] = Field(
        description="A list of reasons why you're compatible with the other person"
    )
    reasons_against: List[str] = Field(
        description="A list of reasons why you're not compatible with the other person"
    )


conversation_parser = PydanticOutputParser(pydantic_object=ConversationSummary)


class CompatibilityEvaluation(BaseModel):
    value_score: int = Field(
        description="A number from 0 to 10 summarizing the compatibility of values between both people"
    )
    personality_score: int = Field(
        description="A number from 0 to 10 summarizing the compatibility of personality between both people"
    )
    hobbies_score: int = Field(
        description="A number from 0 to 10 summarizing the compatibility of hobbies between both people"
    )
    overall_score: int = Field(
        description="A number from 0 to 10 summarizing the overall compatibility between both people"
    )
    reasons_for: List[str] = Field(
        description="A list of reasons why the two people are compatible"
    )
    reasons_against: List[str] = Field(
        description="A list of reasons why the two people are not compatible"
    )


compatibility_parser = PydanticOutputParser(pydantic_object=CompatibilityEvaluation)
