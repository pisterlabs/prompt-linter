import os
import gradio as gr
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI

from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
    LangchainEmbedding,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    set_global_service_context)

from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter



load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_TYPE'] = os.getenv('OPENAI_API_TYPE')
os.environ['OPENAI_API_VERSION'] = os.getenv('OPENAI_API_VERSION')
os.environ['OPENAI_API_BASE'] = os.getenv('OPENAI_API_BASE')

def create_service_context(
    max_input_size:int=4096,
    num_outputs:int=256,
    chunk_overlap_ratio:float=0.1,
    chunk_size_limit:int=None,
    chunk_overlap:int=20,
    chunk_size:int=1024 
    ):

    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        chunk_overlap_ratio,
        chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(
        llm=AzureChatOpenAI(
            # temperature=0.5,
            deployment_name="chatgpt_model",
            max_tokens=num_outputs))

    embedding_llm = LangchainEmbedding(
        langchain_embeddings=OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1)
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=embedding_llm,
        node_parser=node_parser,
        prompt_helper=prompt_helper)

    return service_context



def data_ingestion_indexing(directory_path:str="data"):
    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=create_service_context()
    )

    index.storage_context.persist()

    return index



def data_querying(input_text:str):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    index = load_index_from_storage(
        storage_context, 
        service_context=create_service_context())

    response = index.as_query_engine().query(input_text)
    return response.response



service_context = create_service_context()
set_global_service_context(service_context)
index = data_ingestion_indexing("data")


iface = gr.Interface(
    fn=data_querying,
    inputs=gr.components.Textbox(lines=3, label="Enter your question"),
    outputs="text",
    title="Ask Me Anything About the Chinchilla Paper")

iface.launch(share=False)