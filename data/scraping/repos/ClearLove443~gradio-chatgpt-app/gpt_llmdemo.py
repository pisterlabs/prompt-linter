import os

from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, max_retries=1, model="gpt-4")
gpt_chain = LLMChain(prompt=prompt, llm=llm)
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = gpt_chain.run("Hi")
    print(result)
    print(cb)
