from llama_index import GPTVectorStoreIndex, GPTListIndex
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, LLMPredictor, PromptHelper
from langchain import OpenAI



from embeddings.docparser import getdocument


def create_embeddings(filename : str,filetype:str, current_service_context) :

    document = getdocument("current_active/"+filename,filetype)

    # create index

    vector_index = GPTVectorStoreIndex.from_documents(document,service_context=current_service_context)
    vector_index.storage_context.persist(persist_dir=f"./data/{filename}_vector.json")
    vector_index.storage_context.persist(persist_dir=f"./data/{filename}_vector.json")

    index = GPTListIndex.from_documents(document,service_context=current_service_context)
    index.storage_context.persist(persist_dir=f"./data/{filename}_list.json")

    return vector_index,index


