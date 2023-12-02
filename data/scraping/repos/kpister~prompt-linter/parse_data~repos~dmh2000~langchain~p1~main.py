import sys
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# check that the arguments are correct
if len(sys.argv) < 3:
    print("Usage: python main.py <language> <task>")
    sys.exit(1)

language = sys.argv[1]
task = sys.argv[2]
key = os.getenv("LANGCHAIN")

print("Task:", task)
print("Language:", language)
print("API Key:", key)

llm = OpenAI(openai_api_key=key)

# prompt = f"Write a short {language} program that {task}"
# r = llm(prompt)

prompt = dict([("language", language), ("task", task)])

code_prompt = PromptTemplate(
    template="write a {language} program that {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt)

result = code_chain(prompt)

print(result)
