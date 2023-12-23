from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat_model = ChatOpenAI()

# Simple Prompt Template
prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
print(prompt.format(product="cars"))

# Output parsers
"""
OutputParsers convert raw output of an LLM into a format than can be used downstream. Few types are:
1. Convert text from LLM -> structured data (e.g. JSON)
2. Convert a ChatMessage into just a string
3. Convert the extra information returned from a call besides the message into a string
"""

from langchain.schema import BaseOutputParser 
class CommaSeperatedListOutputParser(BaseOutputParser):
    """
    Parse the output of an LLM call to a comma-seperated list.
    """

    def parse(self, text: str):
        """
        Parse the output of an LLM call
        """
        return text.strip().split(", ")

print(CommaSeperatedListOutputParser().parse("What is, ChatGPT, ?"))