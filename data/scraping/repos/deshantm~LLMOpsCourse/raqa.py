from llama_index import ServiceContext
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()
llm = OpenAI(temperature=0,model="gpt-3.5-turbo",stream=True)
chunk_size = 1000
splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=100)
node_parser = SimpleNodeParser(text_splitter=splitter)

context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=chunk_size
)

from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("llama_index")
chroma_store = ChromaVectorStore(chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=chroma_store)
wiki_vector_index = VectorStoreIndex([],storage_context=storage_context, service_context=context)

from llama_index.readers.wikipedia import WikipediaReader


movie_list = ["Sound of Freedom (film)", "Babylon (2022 film)"]

#fewer auto corrects with auto_suggest=False
reader = WikipediaReader().load_data(pages=movie_list, auto_suggest=False)

for movie, wiki_doc in zip(movie_list, reader):
    #Now we will loop through our documents and metadata and construct nodes (associated with particular metadata for easy filtration later).
    nodes = []
    for document, metadata in zip(wiki_doc, wiki_doc.metadata):
        nodes.append(Node(document=document, metadata=metadata))

    #Now we will add the title metadata to each node.
    for node in nodes:
        node.metadata["title"] = movie
    wiki_vector_index.from_documents(nodes)

  

from llama_index.tools import FunctionTool
from llama_index.vector_stores.types import (
    VectorStoreInfo,
    MetadataInfo,
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from typing import List, Tuple, Any
from pydantic import BaseModel, Field

top_k = 3

vector_store_info = VectorStoreInfo(
    content_info="semantic information about movies",
    metadata_info=[MetadataInfo(
        name="title",
        type="str",
        description="title of movie, one of the movies in the list"
    )]
)

class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(
        ..., description="List of metadata filter field names"
    )
    filter_value_list: List[str] = Field(
        ...,
        description=(
            "List of metadata filter field values (corresponding to names specified in filter_key_list)"
        )
    )
   

def auto_retrieve_fn(
    query: str, filter_key_list: List[str], filter_value_list: List[str]
):
    """Auto retrieval function.

    Performs auto-retrieval from a vector database, and then applies a set of filters.

    """
    query = query or "Query"

    exact_match_filters = [
        ExactMatchFilter(key=k, value=v)
        for k, v in zip(filter_key_list, filter_value_list)
    ]
    retriever = VectorIndexRetriever(
        wiki_vector_index, filters=MetadataFilters(filters=exact_match_filters), top_k=top_k
    )
    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(query)
    return str(response)
    
description = f"""\
Use this tool to look up semantic information about films.
The vector database schema is given below:
{vector_store_info.json()}
"""

auto_retrieve_tool = FunctionTool.from_defaults(
    fn=auto_retrieve_fn,
    name="auto_retrieve",
    description=description,
    fn_schema=AutoRetrieveModel,
)

from llama_index.agent import OpenAIAgent

agent = OpenAIAgent.from_tools(
    tools=[auto_retrieve_tool],
    service_context=context,
    name="llama_index",
    description="A tool for looking up semantic information about films",
    version="0.0.1",
)

response = agent.chat("Tell me what happens (briefly) in the Sound of Freedom movie.")
print(str(response))   

