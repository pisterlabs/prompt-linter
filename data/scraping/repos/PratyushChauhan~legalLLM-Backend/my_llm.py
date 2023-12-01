from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
import os
import keysecrets
# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

os.environ['OPENAI_API_KEY'] = keysecrets.apiKey

# model_path = "llama-2-13b-chat.Q4_0.gguf"
model_path = "llama-2-7b-chat.Q5_K_M.gguf"
def get_llm():
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=None,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=2048,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 34},  #28,29,30 layers works best on my setup.
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


def get_query_engine(llm):
    # use Huggingface embeddings
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    # create a service context
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    # load documents
    documents = SimpleDirectoryReader(
        "./docs/agreements"
    ).load_data()
    # create vector store index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # set up query engine
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=1
    )
    return query_engine

def openAI_queryEngine():
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    # create a service context
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
    )
    # load documents
    documents = SimpleDirectoryReader(
        "./docs/agreements"
    ).load_data()
    # create vector store index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # set up query engine
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=1
    )
    return query_engine