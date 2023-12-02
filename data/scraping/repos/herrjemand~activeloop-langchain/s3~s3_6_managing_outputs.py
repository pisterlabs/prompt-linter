from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on the context")

    @validator('words')
    def not_start_with_number(cls, v):
        for word in v:
            if word[0].isnumeric():
                raise ValueError('word should not start with a number')
        return v
    
parser = PydanticOutputParser(pydantic_object=Suggestions)

from langchain.prompts import PromptTemplate

template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context.
{format_instructions}

target_word: {target_word}
context: {context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['target_word', 'context'],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)


from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)
output = llm(model_input.to_string())
result = parser.parse(output)

print(result)