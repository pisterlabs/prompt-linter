"""
https://python.langchain.com/docs/modules/chains/foundational/router

本笔记本演示了如何使用 RouterChain 范例创建一个链，该链 动态选择 用于给定输入的下一个链。

two components:
1. The RouterChain itself (responsible for selecting the next chain to call)
2. destination_chains: 路由器链可以路由到的链
"""

import env

from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# 物理学家
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


# 数学家
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

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

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

print(chain.run("What is black body radiation?"))

print(
    chain.run(
        "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3"
    )
)


print(chain.run("What is the name of the type of cloud that rins"))

"""
EmbeddingRouterChain
The EmbeddingRouterChain uses embeddings and similarity to route between destination chains.
"""

from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma

names_and_descriptions = [
    ("physics", ["for questions about physics"]),
    ("math", ["for questions about math"]),
]

router_chain = EmbeddingRouterChain.from_names_and_descriptions(
    names_and_descriptions, Chroma, CohereEmbeddings(), routing_keys=["input"]
)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.run("What is black body radiation?"))



print(
    chain.run(
        "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3"
    )
)