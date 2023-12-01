import logging
import sys

from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from langchain import OpenAI


# define LLM
# NOTE: at the time of demo, text-davinci-002 did not have rate-limit errors
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)


oracle = GPTKnowledgeGraphIndex.load_from_disk('ontology/kg.json', service_context=service_context)

response = oracle.query(
    "What is Plurigrid?",
    include_text=False,
    response_mode="tree_summarize"
)

print(response)
