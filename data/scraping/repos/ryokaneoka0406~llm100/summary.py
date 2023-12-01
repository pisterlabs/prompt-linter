import openai
from dotenv import load_dotenv
import os

import logging
import sys

from llama_index import GPTListIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
        max_tokens=1024,
    )
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    chunk_size_limit=1024
)


documents = SimpleDirectoryReader('docs').load_data()
index = GPTListIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine(
    response_mode='tree_summarize'
)
response = query_engine.query('この文章の要点がわかるようにまとめ、日本語で出力してください。')

print(response)
