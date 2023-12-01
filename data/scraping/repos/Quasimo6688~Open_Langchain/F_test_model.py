import os

import dotenv
import pydantic
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.schema import AgentFinish
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI


script_dir = os.path.dirname(os.path.abspath(__file__))
api_key_file_path = os.path.join(script_dir, 'key.txt')

# 自动填写OpenAI API
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")

llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

primes = {998: 7901, 999: 7907, 1000: 7919}


class CalculatorInput(pydantic.BaseModel):
    question: str = pydantic.Field()


class PrimeInput(pydantic.BaseModel):
    n: int = pydantic.Field()


def is_prime(n: int) -> bool:
    if n <= 1 or (n % 2 == 0 and n > 2):
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_prime(n: int, primes: dict = primes) -> str:
    return str(primes.get(int(n)))


async def aget_prime(n: int, primes: dict = primes) -> str:
    return str(primes.get(int(n)))


tools = [
    Tool(
        name="GetPrime",
        func=get_prime,
        description="A tool that returns the `n`th prime number",
        args_schema=PrimeInput,
        coroutine=aget_prime,
    ),
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to compute mathematical expressions",
        args_schema=CalculatorInput,
        coroutine=llm_math_chain.arun,
    ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

question = "What is the product of the 998th, 999th and 1000th prime numbers?？"

for step in agent.iter(question):
    if output := step.get("intermediate_step"):
        action, value = output[0]
        if action.tool == "GetPrime":
            print(f"Checking whether {value} is prime...")
            assert is_prime(int(value))
        # Ask user if they want to continue
        _continue = input("Should the agent continue (Y/n)?:\n")
        if _continue != "Y":
            break

