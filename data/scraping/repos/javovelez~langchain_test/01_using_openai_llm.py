import os
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = os.getenv('ACTIVELOOP_TOKEN')


llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)
text = "say hello in japanese, give me the shortest answer that you can."
print(llm(text))

