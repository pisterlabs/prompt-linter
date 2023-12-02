from typing import List
import re

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)

# this is NOT desciptive enough for the model to understand what we want it to do.
# class Name(BaseModel):
#     first_name: str = Field(description="the first name of the person")
#     last_name: str = Field(description="the last name of the person")


# We can use a docstring to explain the whole class, which gives us the right output
# class Name(BaseModel):
#     "the name of the person returned as a result of the user query."
#     first_name: str = Field(description="the first name of the person")
#     last_name: str = Field(description="the last name of the person")


# Or we can simply be more descriptive with our class attributes - ideally both!
class Name(BaseModel):
    first_name: str = Field(
        description="the first name of the person returned as an answer to the user's query"
    )
    last_name: str = Field(
        description="the last name of the person returned as an answer to the user's query"
    )

    # We can also use custom validation logic!
    # this means that this validation function will be used
    @validator("first_name", "last_name")
    def name_is_valid(cls, field):
        pattern = r"^[A-Za-z\-\'\s]+$"
        if not re.match(pattern, field):
            raise ValueError("Name contains invalid characters")
        return field


# query = "Answer the user query: \n {format_instructions} \n who discovered the theory of relativity?"
query = "Answer the user query: \n {format_instructions} \n what is the name of the millitary leader \
         who took taiwan in 1945? Please return their name in chinese characters"


parser = PydanticOutputParser(pydantic_object=Name)
print(parser.get_format_instructions())

prompt = ChatPromptTemplate.from_messages([query])
input = prompt.format_prompt(
    query=query,
    format_instructions=parser.get_format_instructions(),
).to_messages()

# As we can see, the get_format_instructions inserts a tried and tested prompt
# to help the model understand what we want it to do.
# print(input)

output = model(input).content
result = parser.parse(output)

print(result)
# We can see that it returns a name Object inside of Python. How neat!
print(type(result))
