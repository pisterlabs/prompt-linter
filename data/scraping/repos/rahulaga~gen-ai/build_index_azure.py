import os
from llama_index import LLMPredictor, VectorStoreIndex, SimpleDirectoryReader, ServiceContext, LangchainEmbedding
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
import openai
import logging
import sys

#llamaindex logs
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#Enable to show openai logs
#openai.log='debug'

#Based on your settings, see version, base, key in your Azure AI portal
api_type = "azure"
api_version = "2023-03-15-preview"
api_base = os.getenv("AZURE_API_BASE")
api_key = os.getenv("AZURE_API_KEY")
chat_deployment = "gpt35"
embedding_deployment= "text-embedding-ada-002"

# Chat model
llm = AzureOpenAI(deployment_name=chat_deployment, openai_api_base=api_base, openai_api_key=api_key, model_kwargs={
    "api_type": api_type,
    "api_version": api_version,
})
llm_predictor = LLMPredictor(llm=llm)

# Embedding model
embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model=embedding_deployment,
        deployment=embedding_deployment,
        openai_api_key=api_key,
        openai_api_base=api_base,
        openai_api_type=api_type,
        openai_api_version=api_version,
    ),
    embed_batch_size=1
)

#load docs
documents = SimpleDirectoryReader('local-data').load_data()

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embedding_llm)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

index.storage_context.persist(persist_dir="local-index-azure")
print("Saved embeddings")
