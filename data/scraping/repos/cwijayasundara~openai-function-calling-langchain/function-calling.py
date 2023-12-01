import os
import openai

from typing import Optional
from langchain.chains.openai_functions import (
    create_structured_output_chain
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from typing import Sequence


class Person(BaseModel):
    """Identifying information about a person."""
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")


# If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
os.environ['OPENAI_API_KEY'] = ""
openai.api_key = os.environ['OPENAI_API_KEY']
llm_name = "gpt-4"
llm = ChatOpenAI(model_name=llm_name, temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a world class algorithm for extracting information in structured formats."),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

chain = create_structured_output_chain(Person, llm, prompt, verbose=True)
response = chain.run("Sally is 13")
print(response)


class People(BaseModel):
    """Identifying information about all people in a text."""
    people: Sequence[Person] = Field(..., description="The people in the text")

chain = create_structured_output_chain(People, llm, prompt, verbose=True)

response = chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally."
)
print(response)

# use a json schema

json_schema = {
    "title": "Person",
    "description": "Identifying information about a person.",
    "type": "object",
    "properties": {
        "name": {"title": "Name", "description": "The person's name", "type": "string"},
        "age": {"title": "Age", "description": "The person's age", "type": "integer"},
        "fav_food": {
            "title": "Fav Food",
            "description": "The person's favorite food",
            "type": "string",
        },
    },
    "required": ["name", "age"],
}
chain = create_structured_output_chain(json_schema, llm, prompt, verbose=True)
response = chain.run("Sally is 13")
print('from json schema', response)
