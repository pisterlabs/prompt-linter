import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

    # Simple Chain
    # The most elementary type of chain is known as a basic chain, which represents the simplest form of crafting a chain. In this setup, there is only one LLM responsible for receiving an input prompt and using it for generating text.

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI()
prompt = PromptTemplate(input_variables=["place"],
                        template="Best place to visit place in {place} ?")

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("Ahmedabad"))

    # Simple sequential chain
    # Sequential chains involves making a series of consecutive calls to the language model. This approach proves especially valuable when there is a need to utilize the output generated from one call as input for another call.

from langchain.chains import SimpleSequentialChain
from langchain.llms import HuggingFaceHub

template = """You have to suggest 5 best places to visit in {place}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["place"], template=template)

HF_LLM = HuggingFaceHub(repo_id = "google/flan-t5-xxl")
place_chain = LLMChain(llm=HF_LLM, prompt=prompt_template)

template = """Given a list of places, Please estimate the expense to visit all of them in local currency and also the days needed in {expenses}

YOUR RESPONSE:
"""
prompt_template = PromptTemplate(input_variables=["expenses"], template=template)

llm = OpenAI()
expense_chain = LLMChain(llm=llm, prompt=prompt_template)
final_chain = SimpleSequentialChain(chains=[place_chain, expense_chain], verbose=True)
review = final_chain.run("Mumbai")


