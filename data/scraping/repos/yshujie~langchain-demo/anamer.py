import asyncio 
import time 

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_serially():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=['product_name'],
        template="What is a good name for a company that makes {product_name}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    for _ in range(3):
        resp = chain.run("cars")
        print(resp)
    
async def async_generate(chain):
    resp = await asyncio.to_thread(chain.run, product_name="cars")
    print(resp)
    
async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=['product_name'],
        template="What is a good name for a company that makes {product_name}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    tasks = [async_generate(chain) for _ in range(3)]
    await asyncio.gather(*tasks)
    