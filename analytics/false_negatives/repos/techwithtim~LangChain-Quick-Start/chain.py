from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(openai_api_key=api_key)

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chain = chat_prompt | chat_model | CommaSeparatedListOutputParser()
result = chain.invoke({"text": "colors"})
print(result)