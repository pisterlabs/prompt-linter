from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


class Step(BaseModel):
    order: int = Field(description="number of the step")
    subtitle: str = Field(description="title of the step")
    desc: str = Field(description="brief description of the step")

    # TODO: Add validators to control the length of the output


class Roadmap(BaseModel):
    title: str = Field(description="title of the roadmap")
    steps: List[Step] = Field(description="steps needed to achieve the goal")

    # # You can add custom validation logic easily with Pydantic.
    # @validator("setup")
    # def question_ends_with_question_mark(cls, field):
    #     if field[-1] != "?":
    #         raise ValueError("Badly formed question!")
    #     return field






class StepDetails(BaseModel):
    content: str = Field(description="detailed guide with minimum of 250 words")
    resources: List[str] = Field(description="List of Helpful links to help understand the step, must be valid URLs")
    # people: List[str] = Field(description="List of names of people who are good in this profession")
    keywords: List[str] = Field(description="List of keywords that highlights the most important aspects in this step ",max_items=5)
    # TODO: Add validators for the URLs of the resources and people
