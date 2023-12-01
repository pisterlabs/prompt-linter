import langchain
from langchain.llms import OpenAI
import os

# insert API_TOKEN in the file to be read here
with open('Temp/open_ai_token.txt', 'r') as file:
    os.environ["OPENAI_API_KEY"] = file.read().rstrip()

# Instanciating a proprietary LLM from OpenAI
llm = OpenAI(model_name="text-davinci-003")

# The LLM takes a prompt as an input and outputs a completion
# prompt = "Alice has a parrot. What animal is Alice's pet?"
# completion = llm(prompt)