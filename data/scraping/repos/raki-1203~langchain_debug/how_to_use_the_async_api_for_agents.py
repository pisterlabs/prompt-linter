import os
import time
import asyncio

from aiohttp import ClientSession

from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.llms import OpenAI
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY


async def main():
    async with ClientSession() as session:
        tasks = [agent.arun(q) for q in questions]
        return await asyncio.gather(*tasks)


if __name__ == '__main__':
    questions = [
        "Who won the US Open men's final in 2019? What is his age raised to the 0.334 poser?",
        "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?",
        "Who won the most recent formula 1 grand prix? What is their age raised to the 0.23 power?",
        "Who won the US Open women's final in 2019? What is her age raised to the 0.34 power?",
        "Who is beyonce's husband? What is his age raised to the 0.19 power?",
    ]

    llm = OpenAI(temperature=0)
    tools = load_tools(["google-serper", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    s = time.perf_counter()
    for q in questions:
        agent.run(q)
    elapsed = time.perf_counter() - s
    print(f"Serial executed in {elapsed:0.2f} seconds.")  # 68.69 seconds

    s = time.perf_counter()
    # If running this outside of Jupyter, use asyncio.run or loop.run_until_complete
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"Concurrent executed in {elapsed:0.2f} seconds.")  # 17.37 seconds


