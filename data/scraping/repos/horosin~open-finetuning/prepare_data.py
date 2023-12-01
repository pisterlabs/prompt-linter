import os
import openai
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import json


openai.api_key = os.getenv("OPENAI_API_KEY")

example = {
    "name": "John Doe",
    "handle": "jdoe",
    "age": 30,
    "hobbies": ["programming", "gaming", "reading"],
    "email": "john.doe@gmail.com",
    "bio": "Avid Python programmer and gamer",
    "location": "USA",
    "is_blue_badge": True,
    "joined": "2019-01-01",
    "gender": "male",
    "appearance": "white, brown hair, brown eyes, 6'0",
    "avatar_prompt": "A photorealistic image of a man, white, brown hair, brown eyes, against a blurred backdrop od a home scenery. Use a Hasselblad camera with a 85mm lens at F 1.2 aperture setting and soft sunlight falling on the subject to capture the subject’s creativity.",
    "banner_prompt": "High resolution photo of a desk with a laptop, a gamepad and a book. Canon DSLR.",
}


# Define a new Pydantic model with field descriptions and tailored for Twitter.
class TwitterUser(BaseModel):
    name: str = Field(description="Full name of the user.")
    handle: str = Field(description="Twitter handle of the user, without the '@'.")
    age: int = Field(description="Age of the user.")
    hobbies: List[str] = Field(description="List of hobbies of the user.")
    email: str = Field(description="Email address of the user.")
    bio: str = Field(description="Bio or short description about the user.")
    location: str = Field(description="Location or region where the user resides.")
    is_blue_badge: bool = Field(
        description="Boolean indicating if the user has a verified blue badge."
    )
    joined: str = Field(description="Date the user joined Twitter.")
    gender: str = Field(description="Gender of the user.")
    appearance: str = Field(description="Physical description of the user.")
    avatar_prompt: str = Field(
        description="Prompt for generating a photorealistic avatar image. The image should capture the essence of the user's appearance description, ideally in a setting that aligns with their interests or bio. Use professional equipment to ensure high quality and fine details."
    )
    banner_prompt: str = Field(
        description="Prompt for generating a banner image. This image should represent the user's hobbies, interests, or the essence of their bio. It should be high-resolution and captivating, suitable for a Twitter profile banner."
    )


# Set up a query tailored for Twitter.
user_query = "Generate a detailed Twitter profile of a random realistic user with a diverse background, from any country in the world, original name, including prompts for images. Come up with real name, never use most popular placeholders like john smith and john doe."

# Instantiate the parser with the new model.
parser = PydanticOutputParser(pydantic_object=TwitterUser)

# Update the prompt to match the new query and desired format.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\nExample:\n{example}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
        "example": json.dumps(example),
    },
)

# Generate the input using the updated prompt.
_input = prompt.format_prompt(query=user_query)

print(_input.to_string())


prompt = """
Answer the user query.
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"name": {"description": "Full name of the user.", "title": "Name", "type": "string"}, "handle": {"description": "Twitter handle of the user, without the '@'.", "title": "Handle", "type": "string"}, "age": {"description": "Age of the user.", "title": "Age", "type": "integer"}, "hobbies": {"description": "List of hobbies of the user.", "items": {"type": "string"}, "title": "Hobbies", "type": "array"}, "email": {"description": "Email address of the user.", "title": "Email", "type": "string"}, "bio": {"description": "Bio or short description about the user.", "title": "Bio", "type": "string"}, "location": {"description": "Location or region where the user resides.", "title": "Location", "type": "string"}, "is_blue_badge": {"description": "Boolean indicating if the user has a verified blue badge.", "title": "Is Blue Badge", "type": "boolean"}, "joined": {"description": "Date the user joined Twitter.", "title": "Joined", "type": "string"}, "gender": {"description": "Gender of the user.", "title": "Gender", "type": "string"}, "appearance": {"description": "Physical description of the user.", "title": "Appearance", "type": "string"}, "avatar_prompt": {"description": "Prompt for generating a photorealistic avatar image. The image should capture the essence of the user's appearance description, ideally in a setting that aligns with their interests or bio. Use professional equipment to ensure high quality and fine details.", "title": "Avatar Prompt", "type": "string"}, "banner_prompt": {"description": "Prompt for generating a banner image. This image should represent the user's hobbies, interests, or the essence of their bio. It should be high-resolution and captivating, suitable for a Twitter profile banner.", "title": "Banner Prompt", "type": "string"}}, "required": ["name", "handle", "age", "hobbies", "email", "bio", "location", "is_blue_badge", "joined", "gender", "appearance", "avatar_prompt", "banner_prompt"]}
```
Example:
{"name": "John Doe", "handle": "jdoe", "age": 30, "hobbies": ["programming", "gaming", "reading"], "email": "john.doe@gmail.com", "bio": "Avid Python programmer and gamer", "location": "USA", "is_blue_badge": true, "joined": "2019-01-01", "gender": "male", "appearance": "white, brown hair, brown eyes, 6'0", "avatar_prompt": "A photorealistic image of a man, white, brown hair, brown eyes, against a blurred backdrop od a home scenery. Use a Hasselblad camera with a 85mm lens at F 1.2 aperture setting and soft sunlight falling on the subject to capture the subject\u2019s creativity.", "banner_prompt": "High resolution photo of a desk with a laptop, a gamepad and a book. Canon DSLR."}
Generate a detailed Twitter profile of a random realistic user with a diverse background, from any country in the world, original name, including prompts for images. Come up with real name, never use most popular placeholders like john smith and john doe.

"""
# create a chat completion

SAMPLE_COUNT = 1

examples = []

for i in range(SAMPLE_COUNT):
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    print(chat_completion.choices[0].message.content)
    examples.append(json.loads(chat_completion.choices[0].message.content))

with open("training_examples.json", "w") as outfile:
    outfile.write(json.dumps(examples))
