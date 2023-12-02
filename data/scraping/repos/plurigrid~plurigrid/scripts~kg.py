import logging
import sys
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from langchain import OpenAI

from llama_index import GPTSimpleVectorIndex, download_loader


WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
wikipedia = loader.load_data(pages=['transactive energy'])

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

gm = SimpleDirectoryReader('ontology').load_data()

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

kg = GPTKnowledgeGraphIndex.from_documents(
    gm,
    max_triplets_per_chunk=2,
    service_context=service_context
)

kg.save_to_disk('ontology/kg.json')
