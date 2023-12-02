# Langchain Examples - PromptTemplate + Chat LLM + OutputParser
# References: https://python.langchain.com/docs/get_started/quickstart#prompttemplate--llm--outputparser

#%%

import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

from dotenv import load_dotenv



#%%
# Define a prompt template (input)

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])


#%%
# Define a parser for the output 

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


#%%
# Define the model & connect to it

load_dotenv()

chat = ChatOpenAI(openai_api_key = os.getenv('openai_organization_key'))


#%%
# Define the chain & run it

chain = chat_prompt | chat | CommaSeparatedListOutputParser()
chain.invoke({"text": "colors"})


# %%
