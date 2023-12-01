import logging
import sys
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import argparse
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, NotionPageReader,  SimpleDirectoryReader, LLMPredictor, GPTListIndex, readers, PromptHelper

def build_index(document_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2048
    # set maximum chunk overlap
    max_chunk_overlap = 256
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"), max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(document_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index

def query_index(query, index_path):
    index = GPTSimpleVectorIndex.load_from_disk(index_path)
    response = index.query(query, mode="embedding", similarity_top_k=3, response_mode="compact")
    print(response)

def main():

    # Load the .env file
    load_dotenv()

    query = "You are an expert personal decision advisor, and the text is my journal entries. I will tell you about my thoughts. Your task is to make comments that help me avoid thought traps and biases. Point out if I already wrote about similar things, if my reasoning aligns with the values I expressed in my writing, and if my reasoning shows any bias. Quote relevant passages from my writing in the original language.\n\n"
    user_input = input("My thoughts: ")
    query = query + user_input

    index = build_index("data/advisor")
    query_index(query, "index.json")

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    time_elapsed = end_time - start_time

    print("Time elapsed: {:.2f} minutes".format(time_elapsed / 60))