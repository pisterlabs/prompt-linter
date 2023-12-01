import os
import textwrap

from llama_index import GPTVectorStoreIndex, LLMPredictor, SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.evaluation import QueryResponseEvaluator
from retrieve_doc_nodes import ingest_main
from discord_reader import hit_discord
import openai
from langchain.embeddings.cohere import CohereEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from dotenv import load_dotenv
from llama_index.embeddings import OpenAIEmbedding

embed_model = OpenAIEmbedding()

load_dotenv()

cohere_api_key = os.environ.get("COHERE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
activeloop_key = os.environ.get("ACTIVELOOP_TOKEN")

os.environ["OPENAI_API"] = openai_api_key
os.environ[
    "ACTIVELOOP_TOKEN"
] = activeloop_key
openai.api_key = openai_api_key

service_context = ServiceContext.from_defaults(embed_model=embed_model)
# Below is for Medium, input Slug in argument
# documents = process_medium()

# Below is for documentation, pass in an ARRAY of top level documents. I.e ["https://docs.example.com"]
#documents = ingest_main(["https://docs.solana.com/"])

# For Discord
documents = hit_discord()

# Below is for code repos
# documents = retrieve_repo_docs("bal_repos")

# Below is for Custom Docs
# documents = create_documents_from_csv('tali-updated-5/balancer_custom_ingestion.csv')
print("docs length", len(documents))
dataset_path = "hub://tali/ocean_protocol_discord"

# Create an index over the documnts
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)
