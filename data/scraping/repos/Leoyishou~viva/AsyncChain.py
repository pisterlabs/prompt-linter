import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "886ab329f3d0dda244f3544efeb257cc077d297bb0c666f5c76296d25c0b2279"
import asyncio
import time

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def generate_serially():
    llm = OpenAI(temperature=0.1)
    prompt = PromptTemplate(
        input_variables=["product"],
        template=" {a} 和  {b} 的关系是什么?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run(a="朱元璋", b="朱允炆")
    print(resp)
    resp = chain.run(a="武则天", b="李治")
    print(resp)

async def async_generateB(chain):
    resp = await chain.arun(a="武则天", b="李治")
    print(resp)

async def async_generateA(chain):
    resp = await chain.arun(a="朱元璋", b="朱允炆")
    print(resp)

async def generate_concurrently():
    llm = OpenAI(temperature=0.1)
    prompt = PromptTemplate(
        input_variables=["a", "b"],
        template=" {a} 和  {b} 的关系是什么?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tasks = [async_generateA(chain) , async_generateB(chain)]
    await asyncio.gather(*tasks)




s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
asyncio.run(generate_concurrently())
elapsed = time.perf_counter() - s
print("\033[1m" + f"异步调用的时间 in {elapsed:0.2f} seconds." + "\033[0m")


s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print("\033[1m" + f"线性调用的时间 in {elapsed:0.2f} seconds." + "\033[0m")