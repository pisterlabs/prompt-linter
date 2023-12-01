from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Optional
import re
from .helpers.utils import (
    instruction_response,
    output_format_checker,
    generate_full_prompt,
    retry_on_rate_limit_error,
)
from .helpers.create_logger import create_logger
import ast

load_dotenv()
import os
import time

LOG = create_logger()


@retry_on_rate_limit_error(wait_time=30)
@output_format_checker(max_attempts=10, desired_format=tuple)
def retry_underlyings(
    document_path: str,
    prev_underlyings: Optional[List[str]],
    prev_strike: Optional[List[float]],
) -> tuple[List[str], List[float]]:
    with open(document_path, "r") as f:
        loaded_json = json.load(f)
    documents = loaded_json["full_text"]
    text_splitter = CharacterTextSplitter(
        chunk_size=800, chunk_overlap=0.2, separator=" "
    )
    texts = text_splitter.split_text(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    underz = str(prev_underlyings)
    strikez = str(prev_strike)
    docs = retriever.get_relevant_documents(
        underz,
    )
    docs_strike = retriever.get_relevant_documents(
        strikez,
    )
    prompt = PromptTemplate.from_template(
        "Extract the tickers of the underlyings (components/stocks) of term sheet"
        "The strike prices and underlyings are usually mentioned together, and should be lists of the same length."
        "We have the following underlyings: {underlyings}"
        "We have the following strike prices: {strike}"
        "Return me the list of underlyings and strike prices in the format of:\n"
        "underlyings: [ticker,ticker,...]\n"
        "strike: [float,float,...]"
        "The lists have to be the same length and should not repeat any tickers or floats"
        "Your response format example: [ticker,ticker,...],[float,float,...]"
    )
    context = ""
    for doc in docs:
        context += "\n" + doc.page_content
    for doc in docs_strike:
        context += "\n" + doc.page_content
    context = prompt.format(underlyings=underz, strike=strikez)
    result = instruction_response(context)
    pattern = r"\[[^\]]*\]"
    # Find all matches
    matches = re.findall(pattern, result)
    if len(matches) == 2:
        # extract the two lists and identify which list has floats and which has strings
        list1 = matches[0]
        list2 = matches[1]
        list1 = ast.literal_eval(list1)
        list2 = ast.literal_eval(list2)
        if all(isinstance(item, str) for item in list1):
            # list1 is the list of underlyings
            underlyings = list1
            strike = list2
            return underlyings, strike
        else:
            # list2 is the list of underlyings
            underlyings = list2
            strike = list1
        return underlyings, strike


@retry_on_rate_limit_error(wait_time=10)
@output_format_checker(max_attempts=10, desired_format=list)
def get_underlyings(document_path: str) -> Optional[List[str]]:
    with open(document_path, "r") as f:
        loaded_json = json.load(f)

    documents = loaded_json["full_text"]
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0.2, separator=" "
    )
    texts = text_splitter.split_text(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    bloomberg = retriever.get_relevant_documents(
        "Bloomberg",
    )
    bbg = retriever.get_relevant_documents(
        "BBG",
    )
    code = retriever.get_relevant_documents(
        "Underlings Index Code",
    )

    prompt = PromptTemplate.from_template(
        "Extract the tickers of the underlyings (components/stocks) of term sheet"
        "The list of underlyings is a list of stocks that the term sheet will include in its investment portfolio"
        "You need to extract the tickers of the underlyings (components/stocks) of term sheet"
        "Make sure that the tickers are presented in the format of tickers"
        "Now we need to make sure that the list of tickers is in the format of tickers"
        "Given the following context:"
        "Context: {context}"
        "Extract the tickers of the underlyings (components/stocks) of term sheet"
        "Your response format example: [ticker,ticker,...]"  # here is the list you asked for : [ticker,ticker,...]
    )

    full_prompt_bloomberg = generate_full_prompt(bloomberg, prompt)
    full_prompt_bbg = generate_full_prompt(bbg, prompt)
    full_prompt_code = generate_full_prompt(code, prompt)

    result_bbg = instruction_response(full_prompt_bbg)
    result_bloomberg = instruction_response(full_prompt_bloomberg)
    result_code = instruction_response(full_prompt_code)

    list_of_results = [result_bbg, result_bloomberg, result_code]
    # now we do a second round of checking

    final_check_template = PromptTemplate.from_template(
        "The list of underlyings is a list of stocks that the term sheet will include in its investment portfolio"
        "You found the following results in the first analysis"
        "Now we need to make sure that the list of tickers is in the format of tickers"
        "You fund the following results for the underlyings: {first_result}"
        "Now we need to make sure that we create a lsit of tickers (acrynoms or codes) of each stock."
        "Your response should be a list"
        "Example of your response: [ticker,ticker,...]"
    )

    final_check = final_check_template.format(first_result=list_of_results)

    final_result = instruction_response(final_check)

    pattern = r"\[[^\]]*\]"
    # Find all matches
    matches = re.findall(
        pattern, final_result
    )  # final_result = ' here is the list you asked for : [ticker,ticker,...]'

    if len(matches) == 1:
        final_result = matches[0]
        final = ast.literal_eval(final_result)
        return final

    else:
        LOG.info("No matches found")

    return final_result
