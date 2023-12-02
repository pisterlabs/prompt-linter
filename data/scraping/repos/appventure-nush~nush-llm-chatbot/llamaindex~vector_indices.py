# initialize simple vector indices + global vector index
import os
import openai
from llama_index import ServiceContext, StorageContext, VectorStoreIndex, SimpleDirectoryReader, \
    load_index_from_storage, LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

package_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(package_dir, "apikey"), "r") as f:
    openai.api_key = f.read().strip()
    # sometimes env bugs out and complains the api_key is not set, so setting it as environment variable just to
    # be safe
    os.environ["OPENAI_API_KEY"] = openai.api_key

if __name__ == "__main__":

    # by default, LlamaIndex uses OpenAI's embedding, we will use HuggingFace's embedding instead since that is free
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    # initialize simple vector indices + global vector index
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index_set = {}
    for chapter in os.listdir(os.path.join(package_dir, "modules/PC3131")):
        docs = SimpleDirectoryReader(os.path.join(package_dir, "modules/PC3131", chapter)).load_data()
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            docs,
            service_context=service_context,
            storage_context=storage_context,
        )
        index_set[chapter] = cur_index
        storage_context.persist(persist_dir=os.path.join(package_dir, f"storage/{chapter}"))
        print("Loaded", chapter)


def load_indices():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    # by default, LlamaIndex uses OpenAI's embedding, we will use HuggingFace's embedding instead since that is free
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    # initialize simple vector indices + global vector index
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    # load indices from disk
    index_set = {}
    for chapter in os.listdir(os.path.join(package_dir, "modules/PC3131")):
        storage_context = StorageContext.from_defaults(persist_dir=os.path.join(package_dir, f"storage/{chapter}"))
        cur_index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
        index_set[chapter] = cur_index
    return index_set
