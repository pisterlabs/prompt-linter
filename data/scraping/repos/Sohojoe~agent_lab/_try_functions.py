import json
from typing import Any, Callable, Optional, Sequence
from altair import Dict
from pydantic import BaseModel, Field

from langchain_helper import convert_pydantic_to_openai_function, create_instance_from_response

class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

    def invoke(self):
        print(f"Invoked Person with name: {self.name}, age: {self.age}, fav_food: {self.fav_food}")

class People(BaseModel):
    """Identifying information about all people in a text."""

    people: Sequence[Person] = Field(..., description="The people in the text")

    def invoke(self):
        print("Invoking People class...")
        for person in self.people:
            person.invoke()    

class RecordPerson(BaseModel):
    """Record some identifying information about a pe."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

    def invoke(self):
        print(f"Recording person with name: {self.name}, age: {self.age}, fav_food: {self.fav_food}")

class RecordDog(BaseModel):
    """Record some identifying information about a dog."""

    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: Optional[str] = Field(None, description="The dog's favorite food")

    def invoke(self):
        print(f"Recording dog with name: {self.name}, color: {self.color}, fav_food: {self.fav_food}")

    

functions = [
    Person,
    People,
    RecordPerson,
    RecordDog
]

openai_functions = [convert_pydantic_to_openai_function(f) for f in functions]
fn_names = [oai_fn["function"]["name"] for oai_fn in openai_functions]


for function in fn_names:
    print(function)


import os
from openai import OpenAI

# model_id = "gpt-3.5-turbo"
# model_id = "gpt-4"
model_id = "gpt-3.5-turbo-1106"
# model_id = "gpt-4-1106-preview"
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  api_key=api_key
)

def get_messages(input):
    messages = [
        # {"role": "system", "content": f"You are a world class algorithm for recording entities."},
        # {"role": "user", "content": f"Make calls to the relevant function to record the entities in the following input: {input}"},
        # {"role": "user", "content": f"Tip: Make sure to answer in the correct format"},
        {"role": "system", "content": f"""
You are a world class algorithm for recording entities.
Make calls to the relevant function to record the entities in the given input.
Tip: Make sure to answer in the correct format         
"""},
        {"role": "user", "content": f"{input}"},
    ]
    return messages
def print_output(output):
    if "function_call" not in output:
        print ("-- no function call --")
        print (output)
    else:
        name = output["function_call"]["name"]
        arguments = output["function_call"]["arguments"]
        print (f"function call: {name}, arguments: {arguments}")


messages = get_messages("Harry was a chubby brown beagle who loved chicken")
response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=openai_functions,
                    tool_choice="auto",
                    temperature=1.0,  # use 0 for debugging/more deterministic results
                    stream=False
                )
output =  response.choices[0].message
# print_output(output)
function_instances = create_instance_from_response(output, functions)
for function_instance in function_instances:
    function_instance.invoke()

messages = get_messages("Sally is 13")
response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=openai_functions,
                    tool_choice="auto",
                    temperature=1.0,  # use 0 for debugging/more deterministic results
                    stream=False
                )
output =  response.choices[0].message
# print_output(output)
function_instances = create_instance_from_response(output, functions)
for function_instance in function_instances:
    function_instance.invoke()

messages = get_messages("Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally.")
response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=[openai_functions[1]],
                    tool_choice="auto",
                    temperature=0.0,  # use 0 for debugging/more deterministic results
                    stream=False
                )
output =  response.choices[0].message
# print_output(output)
function_instances = create_instance_from_response(output, functions)
for function_instance in function_instances:
    function_instance.invoke()


messages = get_messages("The most important thing to remember about Tommy, my 12 year old, is that he'll do anything for apple pie.")
response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=openai_functions,
                    tool_choice="auto",
                    temperature=1.0,  # use 0 for debugging/more deterministic results
                    stream=False
                )
output =  response.choices[0].message
# print_output(output)
function_instances = create_instance_from_response(output, functions)
for function_instance in function_instances:
    function_instance.invoke()

print ("-- done --")