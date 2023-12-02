#!/usr/bin/env python

import os

import openai
from langchain.chat_models import ChatOpenAI
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, download_loader

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

link = "https://gpt-index.readthedocs.io/en/latest/"


def construct_index(link):
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

    KnowledgeBaseWebReader = download_loader("KnowledgeBaseWebReader")
    loader = KnowledgeBaseWebReader()
    documents = loader.load_data(
      root_url=link,
      link_selectors=['.article-list a', '.article-list a'],
      article_path='/articles',
      body_selector='.article-body',
      title_selector='.article-title',
      subtitle_selector='.article-subtitle',
      )

    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.storage_context.persist(persist_dir="../index/gpt-index_GPTVectorStoreIndex")

    return index


index = construct_index(link)
