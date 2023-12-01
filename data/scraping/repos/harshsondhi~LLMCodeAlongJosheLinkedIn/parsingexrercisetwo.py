import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser

# output_parser = DatetimeOutputParser()
#
# misformatted = result.content
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain import PromptTemplate
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import DatetimeOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
llm = OpenAI()
chat = ChatOpenAI(openai_api_key=api_key)


class Scientist(BaseModel):
    name: str = Field(description="Name of scientis")
    siscoveries: list = Field(description="Python list of discoveries")


parser = PydanticOutputParser(pydantic_object=Scientist)
human_prompt = HumanMessagePromptTemplate.from_template(
    "{request}\n{format_instructions}"
)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
request = chat_prompt.format_prompt(
    request="Tell me about a famous scientist",
    format_instructions=parser.get_format_instructions(),
).to_messages()
result = chat(request, temperature=0)
print(type(parser.parse(result.content)))
