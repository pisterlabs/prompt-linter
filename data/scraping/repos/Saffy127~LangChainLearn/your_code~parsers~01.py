from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

model_name = 'text-davinci-003'
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# Define your desired data structure
class Joke(BaseModel):
  setup: str = Field(description="questiong to set up a joke")
  punchline: str = Field(description="answer to resolve the joke.")

  # You can add custom validation logic easily with Pydantic.
  @validator('setup')
  def question_ends_with_question_mark(cls, field):
    if field[-1] != '?':
        raise ValueError("Badly formed question!")
    return field
  

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
  template="Answer the user query.\n{format_instructions}\n{query}\n",
  input_variables=["query"],
  partial_variables={"format_instructions": parser.get_format_instructions()}
)

# And a query intented to prompt a language model to populate the dat structure.
joke_query = "Tell me a joke."
_input = prompt.format_prompt(query=joke_query)

output = model(_input.to_strring())
parser.parse(output)




    
