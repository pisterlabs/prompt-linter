import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=openai_api_key)
chat_model = ChatOpenAI(openai_api_key=openai_api_key)

text = "What would be a good company name for a company that makes colorful socks?"
answer = llm.predict(text)
print(answer)

answer = chat_model.predict(text)
print(answer)