from dotenv import load_dotenv

load_dotenv()

from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_chain
from typing import Optional
from langchain.prompts import ChatPromptTemplate
import langchain


class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

    @validator("fav_food")
    def is_valid_food(cls, field):
        raise ValueError("This is not a valid food")


# If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world class algorithm for extracting information in structured formats.",
        ),
        (
            "human",
            "Use the given format to extract information from the following input: {input}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

chain = create_structured_output_chain(
    Person,
    llm,
    prompt,
)

# result = chain.run("Sally is 13")

# print(result)

result = chain.apply(
    [
        {"input": "Sally is 13"},
        {"input": "Angela is 13 and likes pizza"},
    ]
)

print(result)
