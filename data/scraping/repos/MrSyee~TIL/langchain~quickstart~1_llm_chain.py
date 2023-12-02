import os

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


openai_api_key = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(openai_api_key=openai_api_key)

system_template = """
        You are a helpful assistant who generates comma separated lists.
        A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
        ONLY return a comma separated list, and nothing more.
    """
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
# chat_prompt.format_messages(input_language="English", output_language="Korean")

chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    output_parser=CommaSeparatedListOutputParser(),
)

output = chain.run(input_language="English", output_language="Korean", text="animal")
print(output)
print(type(output))
