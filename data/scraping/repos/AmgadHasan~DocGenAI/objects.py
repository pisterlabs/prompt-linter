from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


class Introduction(BaseModel):
    app_name: str = Field(description="app name of mobile application")
    purpose: str = Field(description="purpose of mobile application")
    scope: str = Field(description="scope of mobile app")
    target_user: str = Field(description="target user of mobile app")

# Here's another example, but with a compound typed field.
class OverAllDescription(BaseModel):
    product_perspective: str = Field(description="product_perspective for mopile app")




