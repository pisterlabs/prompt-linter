## KNOWLEDGE GRAPH APPROACH:
## Use Llama Index's Knowledge Graph and Query functions

from memory import Memory
from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.readers import Document
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from langchain import OpenAI
import os
from utils import load

class KnowledgeGraphMemory(Memory):
    def __init__(self, source_text, title):
        super().__init__(source_text)

        # Construct a knowledge graph from the source text
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

        source_doc = Document(source_text)
        file_path = f"{title}_index_kg.json"
        if os.path.isfile(file_path):
            print("file exists")
        else:
            index = GPTKnowledgeGraphIndex.from_documents([source_doc],                 max_triplets_per_chunk=15,
                service_context=service_context
                ,
                include_embeddings=True
                )
            index.save_to_disk(file_path)
        self.title = title

    def query(self, query):
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

        file_path = f"{self.title}_index_kg.json"
        new_index = GPTKnowledgeGraphIndex.load_from_disk(file_path, service_context=service_context)

        # Querying the knowledge graph
        response = new_index.query(
            query,
            include_text=False,
            response_mode="tree_summarize",
        )

        return str(response).strip()

if __name__ == "__main__":
    source_text = load("test.txt")
    memory_test = KnowledgeGraphMemory(source_text, "test")

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer)