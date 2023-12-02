from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


parser = PydanticOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

joke_query = "Tell me a joke."
formatted_prompt = prompt.format_prompt(query=joke_query)

print(formatted_prompt.to_string())
"""
Answer the user query.
The output should be formatted as a JSON instance 
that conforms to the JSON schema below.

As an example, for the schema
{
    "properties": {
        "foo": {
            "title": "Foo",
            "description": "a list of strings",
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "foo"
    ]
} 
the object {"foo": ["bar", "baz"]} is a well-formatted 
instance of the schema. 
The object {"properties": {"foo": ["bar", "baz"]}} is 
not well-formatted.

Here is the output schema:
```
{
    "properties": {
        "setup": {
            "title": "Setup",
            "description": "question to set up a joke",
            "type": "string"
        },
        "punchline": {
            "title": "Punchline",
            "description": "answer to resolve the joke",
            "type": "string"
        }
    },
    "required": [
        "setup",
        "punchline"
    ]
}
```
Tell me a joke.
"""


from langchain.llms import OpenAI
from dotenv import load_dotenv


load_dotenv()
model = OpenAI(model_name="text-davinci-003", temperature=0.0)

output = model(formatted_prompt.to_string())
parsed_joke = parser.parse(output)
print(parsed_joke)
