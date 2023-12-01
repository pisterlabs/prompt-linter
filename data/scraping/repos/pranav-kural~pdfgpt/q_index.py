""" Retrieve Index """

from pathlib import Path
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    download_loader
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import pinecone

from file_utils import download_pdf, clear_source_file_contents, delete_source_file

# import parameters
from params import (
    P_SOURCE_FILE_PATH,
    P_DUMMY_FILE_PATH,
    P_INDEX_NAME,
    P_EMBEDDING_MODEL,
    P_CHAT_MODEL,
    P_CHAT_MAX_TOKENS,
    P_CHAT_TEMPERATURE,
    P_EMBEDDING_MAX_TOKENS,
    P_EMBEDDING_TEMPERATURE
)

# method to create new index - generates embeddings for the document and stores in Pinecone
def create_index(document_url):
    """
    method to create new index - generates embeddings for the document and stores in Pinecone
    :param document_url: URL of PDF document
    """

    # remove source file if it exists
    delete_source_file(file_path=P_SOURCE_FILE_PATH)

    # download the document in the document directory
    download_pdf(url=document_url, file_path=P_SOURCE_FILE_PATH)

    # delete existing Pinecone index
    pinecone.delete_index(P_INDEX_NAME)

    # create index
    # dimensions are for text-embedding-ada-002
    pinecone.create_index("gptindex", dimension=1536, metric="cosine", pod_type="Starter")

    # get index from Pinecone
    p_index = pinecone.Index(P_INDEX_NAME)

    # LLM Predictor
    llm_predictor = LLMPredictor(
        llm=OpenAIEmbeddings(temperature=P_EMBEDDING_TEMPERATURE, model_name=P_EMBEDDING_MODEL, max_tokens=P_EMBEDDING_MAX_TOKENS)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # load documents
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    documents = loader.load_data(file=Path(P_SOURCE_FILE_PATH))

    # below is for ingesting text documents
    # documents = SimpleDirectoryReader(P_SOURCE_DIRECTORY).load_data()

    # create pincone vector store
    vector_store = PineconeVectorStore(
        pinecone_index=p_index
    )

    # attach pinecone vector store to LlamaIndex storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # query index
    q_index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

    # print token utilization for building index
    print('Document indexing token utilization: ', llm_predictor.last_token_usage, end='\n')

    # create chat LLM predictor - may use different model
    chat_llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=P_CHAT_TEMPERATURE, model_name=P_CHAT_MODEL, max_tokens=P_CHAT_MAX_TOKENS)
    )
    chat_service_context = ServiceContext.from_defaults(llm_predictor=chat_llm_predictor)

    return q_index, chat_llm_predictor, chat_service_context

def load_index():
    """
    method to load existing index
    """
    # clear source file
    clear_source_file_contents(P_SOURCE_FILE_PATH)

    # get index from Pinecone
    p_index = pinecone.Index(P_INDEX_NAME)

    # LLM Predictor - uses chat model since only loading index not creating
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=P_CHAT_TEMPERATURE, model_name=P_CHAT_MODEL, max_tokens=P_CHAT_MAX_TOKENS)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # load dummy file (empty file)
    documents = SimpleDirectoryReader(P_DUMMY_FILE_PATH).load_data()

    # create pincone vector store
    vector_store = PineconeVectorStore(
        pinecone_index=p_index
    )

    # attach pinecone vector store to LlamaIndex storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # query index
    q_index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

    # print token utilization for building index
    print('Document indexing token utilization: ', llm_predictor.last_token_usage, end='\n')

    return q_index, llm_predictor, service_context