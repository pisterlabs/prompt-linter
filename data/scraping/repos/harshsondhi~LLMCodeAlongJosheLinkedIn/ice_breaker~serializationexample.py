import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
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

template = "tell me a fact about {planet}"

prompt = PromptTemplate(template=template, input_variables=["planet"])
prompt.save("myprompt.json")

loaded_prompt = load_prompt("myprompt.json")
print(loaded_prompt)
