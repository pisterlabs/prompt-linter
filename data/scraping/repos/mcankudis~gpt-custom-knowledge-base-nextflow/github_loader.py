import os
from typing import Sequence

import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from llama_hub.github_repo import GithubClient, GithubRepositoryReader
from llama_index import (GPTVectorStoreIndex, LLMPredictor, PromptHelper,
                         download_loader)
from llama_index.schema import Document

from config import (branch, concurrent_requests, filter_directories,
                    filter_file_extensions, owner, persist_dir, repo)

load_dotenv()
download_loader("GithubRepositoryReader")
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_documents() -> Sequence[Document]:
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    
    loader = GithubRepositoryReader(
        github_client           = github_client,
        owner                   = owner,
        repo                    = repo,
        filter_directories      = filter_directories,
        filter_file_extensions  = filter_file_extensions,
        concurrent_requests     = concurrent_requests
    )
    
    return loader.load_data(branch = branch)


def index_documents(documents: Sequence[Document]):
    max_input_size      = 4096
    num_outputs         = 512
    chunk_overlap_ratio = 0.05
    chunk_size_limit    = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit)
    
    llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo", max_tokens = num_outputs)
    llm_predictor = LLMPredictor(llm)

    index = GPTVectorStoreIndex.from_documents(documents, prompt_helper=prompt_helper, llm_predictor=llm_predictor)
    index.storage_context.persist(persist_dir = persist_dir)


def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    char_sum = 0
    for document in documents: char_sum += len(document.text)
    print(f"Approx. total number of tokens: {char_sum/4}")

    confirmation = input("Continue? (y/n)")
    if confirmation != "y": return

    index_documents(documents)

main()