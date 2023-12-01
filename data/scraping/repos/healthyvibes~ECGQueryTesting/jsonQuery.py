import os 
from pathlib import Path
import openai

from llama_index import download_loader
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, \
                        LLMPredictor, PromptHelper

from llama_index import (
    KeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage
)

from langchain.chat_models import ChatOpenAI 
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from llama_index import StorageContext, load_index_from_storage, ServiceContext, KeywordTableIndex

#api keys

def index_documents(folder):
    max_input_size    = 4096
    num_outputs       = 512
    max_chunk_overlap = 20
    chunk_size_limit  = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
    llm_predictor = LLMPredictor(
        llm = ChatOpenAI(temperature = 0, 
                         model_name = "text-ada-001", 
                         max_tokens = num_outputs)
        )

    JSONReader = download_loader("JSONReader")
    loader = JSONReader()
    documents = loader.load_data(Path(folder))

    index = GPTVectorStoreIndex.from_documents(
                documents, 
                llm_predictor = llm_predictor, 
                prompt_helper = prompt_helper)

    index.storage_context.persist(persist_dir=".") # save in current directory

#index_documents('./sensortest5.json')

def my_chatGPT_bot(input_text):
    # load the index from vector_store.json
    storage_context = StorageContext.from_defaults(persist_dir=".")
    index = load_index_from_storage(storage_context)

    # create a query engine to ask question
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response

#my_chatGPT_bot('What is all the data for uuid 3?')

print("Running pan....")


max_input_size    = 4096
num_outputs       = 512
max_chunk_overlap = 20
chunk_size_limit  = 600

prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    
llm_predictor = LLMPredictor(
    llm = ChatOpenAI(temperature = 0.7, 
                         model_name = "gpt-3.5-turbo", 
                         max_tokens = num_outputs)
        )

#SimpleCSVReader = download_loader("SimpleCSVReader")
#loader = SimpleCSVReader()
#documents = loader.load_data(file=Path('./sense5.csv'))

JSONReader = download_loader("JSONReader")
loader = JSONReader()
documents = loader.load_data(Path('./sensortest5.json'))

index = GPTVectorStoreIndex.from_documents(
                documents, 
                llm_predictor = llm_predictor, 
                prompt_helper = prompt_helper)

query_engine = index.as_query_engine()

#Give me the timestamps for uuid 3 where RR readings are greater than 12 and where blood glucose is less than 110.
response = query_engine.query("What days have rr readings greater than 12 or bg readings less than 100?")
print(response)


"""
documents = SimpleDirectoryReader('pdffolder').load_data()

print(documents)

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# get response from query
query_engine = index.as_query_engine()

response = query_engine.query("What is the most common value in the blood_glucose column?")
print(response)
"""