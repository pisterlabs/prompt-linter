from langchain.llms import OpenAI

import environ

env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")

llm = OpenAI(temperature=0, openai_api_key=API_KEY)
question = "Which language was used to create ChatGPT?"
print(question, llm(question))
