from core.configs import settings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


async def faiss_pdf_ingestor(file_path: str, index_name: str) -> str:
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
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vector_store.save_local(index_name)





async def faiss_qa(index_name: str, question: str) -> str:
    """[summary]

    Args:
        index_name (str): [description]
        question (str): [description]

    Returns:
        str: [description]
    """
    
    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    new_vectorstore = FAISS.load_local(index_name, embeddings=embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=settings.OPENAI_API_KEY), 
                                     chain_type='stuff', retriever=new_vectorstore.as_retriever())
    res = qa.run(question)

    return res