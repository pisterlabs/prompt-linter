"""
LangChain 通过利用 asyncio 库为链提供异步支持。
"""

import env

import asyncio
import time

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_serially():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    for _ in range(2):
        resp = chain.run(product="toothpaste")
        print(resp)

async def async_generate(chain):
    resp = await chain.arun(product="toothpaste")
    print(resp)


async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tasks = [async_generate(chain) for _ in range(2)]
    print("--begin run asyncio.gather(*tasks)")
    await asyncio.gather(*tasks)

s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print("\033[1m" + f"Serial executed in {elapsed:0.2f} seconds." + "\033[0m")

"""
异步的不好使
Retrying langchain.llms.openai.acompletion_with_retry.<locals>._completion_with_retry in 4.0 seconds as it raised APIConnectionError: Error communicating with OpenAI.
"""

s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
asyncio.run(generate_concurrently())
# await generate_concurrently()
elapsed = time.perf_counter() - s
print("\033[1m" + f"Concurrent executed in {elapsed:0.2f} seconds." + "\033[0m")
