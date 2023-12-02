from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "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" # insert your API_TOKEN here
# llm = OpenAI(model_name="text-davinci-003")
# x = llm("Tell me a joke")
# llm_result = llm.generate(["Tell me a joke and Tell me a poem"]*15)
# print(llm_result.generations[0])

import time
import asyncio


def generate_serially():
    llm = OpenAI(temperature=0.9)
    for _ in range(10):
        resp = llm.generate(["Hello, how are you?"])
        print(resp.generations[0][0].text)


async def async_generate(llm):
    resp = await llm.agenerate(["Hello, how are you?"])
    print(resp.generations[0][0].text)


async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)


s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
asyncio.run(generate_concurrently())
# await generate_concurrently() 
elapsed = time.perf_counter() - s
print('\033[1m' + f"Concurrent executed in {elapsed:0.2f} seconds." + '\033[0m')

s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print('\033[1m' + f"Serial executed in {elapsed:0.2f} seconds." + '\033[0m')