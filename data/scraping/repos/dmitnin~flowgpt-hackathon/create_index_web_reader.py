#!/usr/bin/env python
import json
import os

import openai
from langchain.chat_models import ChatOpenAI
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, SimpleWebPageReader

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY


data_file = "../docs/diablo4/links.json"


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_dataset() -> list:
    json_data = read_json_file(data_file)
    return list(json_data.keys())


def construct_index(data_list):
    max_input_size = 4096
    num_output = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    chunk_overlap_ratio = 0.1

    prompt_helper = PromptHelper(
        max_input_size=max_input_size,
        num_output=num_output,
        max_chunk_overlap=max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
        chunk_overlap_ratio=chunk_overlap_ratio)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_output))

    documents = SimpleWebPageReader(html_to_text=True).load_data(data_list)

    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.storage_context.persist(persist_dir="../index/diablo4_GPTVectorStoreIndex")

    return index


links_set = get_dataset()
index = construct_index(links_set)
