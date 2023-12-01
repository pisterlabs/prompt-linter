from core.configs import settings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import pinecone
import boto3



pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT)



async def pinecone_ingest(index_name: str = None, documents = None, chunk_size = 1000, chunk_overlap=30, separator='\n') -> Pinecone:
    """[summary]

    Args:
        index_name (str): [description]
        documents ([type]): [description]
        chunk_size ([type]): [description]
        chunk_overlap ([type]): [description]
        separator ([type]): [description]

    Returns:
        str: [description]
    """
    
    # check if index_name exists in pinecone
    if not index_name in pinecone.list_indexes():
        # create a new index
        pinecone.create_index(index_name, dimension=1536, metric="euclidean")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n') 
    chunks = text_splitter.split_documents(documents=documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    vector_store = Pinecone.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)

    return vector_store




async def pinecone_pdf_ingestor(file_path: str = None, index_name: str = None) -> Pinecone:
    """[summary]

    Args:
        file_path (str): [description]
        index_name (str): [description]

    Returns:
        str: [description]
    """

    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n') 
    chunks = text_splitter.split_documents(documents=documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    vector_store = Pinecone.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)

    return vector_store



async def pinecone_qa(index_name: str, question: str) -> dict:
    """[summary]

    Args:
        index_name (str): [description]
        question (str): [description]

    Returns:
        str: [description]
    """
    
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    chat = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, 
                      verbose=True, temperature=0)

    qa = RetrievalQA.from_chain_type(llm=chat, 
                                     chain_type='stuff', 
                                     retriever=vector_store.as_retriever(), 
                                     return_source_documents=True)
    
    res = qa({"query": question})

    return res