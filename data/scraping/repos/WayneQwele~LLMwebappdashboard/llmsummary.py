
""" This module contains functions for generating summary descriptions of crypto currency pairs using an LLM. """
import os
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import time

from typing import List, Dict
import asyncio
import toml


# Load the secrets file
secrets = toml.load('.secrets/secrets.toml')

# Set the environment variables
for key, value in secrets.items():
    os.environ[key] = value


async def async_generate_summary(chain: LLMChain, product: str) -> str:
    """
    This function generates a summary description of a crypto currency pair using an LLM.

    Args:
    chain (LLMChain): The LLMChain object to use for generating the summary.
    product (str): The name of the crypto currency pair to generate the summary for.

    Returns:
    str: The generated summary description of the crypto currency pair.
    """
    resp = await chain.arun(product=product)
    return resp


async def generate_summary_concurrently(ticker_pairs: List[str]) -> Dict[str, str]:
    """
    This function generates summary descriptions of multiple crypto currency pairs concurrently using an LLM.

    Args:
    ticker_pairs (List[str]): A list of names of the crypto currency pairs to generate the summaries for.

    Returns:
    Dict[str, str]: A dictionary mapping each crypto currency pair name to its generated summary description.
    """
    llm = OpenAI(temperature=0.1)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="Write a summary description of the crypto currency pair {product} highlighting key attributes and popularity, begin by writing the original name of the crypto currency pair first and then the rest of the description. format response in markdown language.",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tasks = [async_generate_summary(chain, product)
             for product in ticker_pairs]
    responses = await asyncio.gather(*tasks)
    return dict(zip(ticker_pairs, responses))
