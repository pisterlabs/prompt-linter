from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pydantic import validator
from typing import List

# Create an output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted List summary of the article")

    # validating the summary field
    @validator("summary")
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated Summary has less than 3 bullet points") 
        return list_of_lines
    
def get_parser():
    parser = PydanticOutputParser(pydantic_object=ArticleSummary)
    return parser