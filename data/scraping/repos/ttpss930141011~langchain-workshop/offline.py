from dotenv import dotenv_values
import os
from langchain.llms import OpenAI

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003",max_tokens=1024)
result = llm("怎麼評價人工智慧")

print(result)