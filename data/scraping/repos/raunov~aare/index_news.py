from dotenv import find_dotenv, load_dotenv

# from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
# new version replaces GPTSimpleVectorIndex with GPTVectorStoreIndex

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from langchain.llms import PromptLayerOpenAI

load_dotenv(find_dotenv())

OpenAI = PromptLayerOpenAI()

company = "Enefit_Green"

def index_documents(company: str) -> GPTVectorStoreIndex:
    # Step 1: Load documents
    documents = SimpleDirectoryReader(f"documents/{company}").load_data()

    # Step 2: Create the index
    index = GPTVectorStoreIndex.from_documents(documents)

    # Step 3: Save the index to disk
    persist_dir = f"storage/{company}"
    index.storage_context.persist(persist_dir=persist_dir)

    return index

index_documents(company)