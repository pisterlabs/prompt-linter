import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, BaseOutputParser
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chains import LLMChain

load_dotenv()

OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
chat_model = ChatOpenAI()

## Predicting with the LLM

text = "What would be a good company name for a company that makes colorful socks?"

# print(llm.predict(text))
# print(chat_model.predict(text))

## Using the HumanMessage schema

messages = [HumanMessage(content=text)]

# print(llm.predict_messages(messages))
# print(chat_model.predict_messages(messages))

## Prompt Templates (https://python.langchain.com/docs/modules/model_io/prompts)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good company name for a company that makes {product}?")
# print(prompt.format(product="colorful socks"))

## ChatMessageTemplates (https://python.langchain.com/docs/modules/model_io/prompts)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# print(chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming."))
# print(chat_model.predict_messages(chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")))

## Output Parsers (https://python.langchain.com/docs/modules/model_io/output_parsers)

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

# print(CommaSeparatedListOutputParser().parse("Hi, bye"))

## LLMChain

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generated 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=ChatOpenAI(), prompt=chat_prompt, output_parser=CommaSeparatedListOutputParser())

# print(chain.run("colors"))