# https://python.langchain.com/docs/modules/model_io/output_parsers/retry

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser,
    RetryOutputParser,
)
from pydantic import BaseModel, Field, validator
from typing import List

template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:
"""


class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

parser = PydanticOutputParser(pydantic_object=Action)

prompt = ChatPromptTemplate.from_messages([template])

request = prompt.format_prompt(query = "who is leo decaprio's girlfriend?", 
                                format_instructions=parser.get_format_instructions(),).to_messages()

bad_response = '{"action": "search"}'

# parser.parse(bad_response)

# this will cause confusion because the parser does not expect what to put for action_input
# this actually worked, lol
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(model='gpt-3.5-turbo', verbose=True))
print(fix_parser.parse(bad_response))