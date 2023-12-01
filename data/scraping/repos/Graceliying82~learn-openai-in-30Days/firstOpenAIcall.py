import os
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()

llm = OpenAI(openai_api_key = os.getenv("OPENAI_API_KEY"))
llm = OpenAI(temperature=0.9)

text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))