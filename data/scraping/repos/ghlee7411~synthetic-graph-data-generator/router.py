from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

import json
import ast
import click
load_dotenv(verbose=True, override=True)
del load_dotenv


@click.command()
@click.option("--instruction", '-i', default="What is the first prime number greater than 40 such that one plus the prime number is divisible by 3", help="Instruction to run")
def main(instruction: str):

    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise and easy to understand manner. \
    When you don't know the answer to a question you admit that you don't know.

    Here is a question:
    {input}"""


    math_template = """You are a very good mathematician. You are great at answering math questions. \
    You are so good because you are able to break down hard problems into their component parts, \
    answer the component parts, and then put them together to answer the broader question.

    Here is a question:
    {input}"""

    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template,
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": math_template,
        },
    ]

    llm = OpenAI()

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    default_chain = ConversationChain(llm=llm, output_key="text")

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    result = chain.run(instruction)
    
    print('\n[RESULT]')
    print(result)


if __name__ == "__main__":
    main()
