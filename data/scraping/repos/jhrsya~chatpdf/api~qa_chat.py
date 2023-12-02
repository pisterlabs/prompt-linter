from utils.pinecone_tools import init_pinepone, make_sure_index_exist
from langchain.vectorstores import Chroma, Pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from typing import List, Callable
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from utils.constant import CHROMA_PATH
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from utils.pdf_loader import BackPdfLoader


def get_answer_from_chain(docs: List[Document], query: str, load_chain: Callable):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    chain = load_chain(llm, chain_type="stuff")
    answer = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return answer

def get_similar_docs_from_pinecone(index_name: str, documents: List[Document], query: str, model_name: str = "davinci"):
    """从pinecone中获取相似的文档"""
    # 初始化pinecone
    init_pinepone()
    make_sure_index_exist(index_name=index_name)

    embeddings = OpenAIEmbeddings(model=model_name)  # 必须要给出model="davinci"，否则会报错

    docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name)
    docs = docsearch.similarity_search(query)
    return docs

def get_similar_docs_from_chroma(query: str = None, documents: List[Document] = None, 
                                 model_name: str = "davinci",
                                 chroma_directory: str = CHROMA_PATH) -> List[Document]:
    """从chroma中获取相似的文档"""
    # if not os.path.exists(os.path.join(chroma_directory, "index")):
    #     docsearch = create_index_by_chroma(documents=documents, model_name=model_name, 
    #                            chroma_directory=chroma_directory)
    # else:
    docsearch = read_index_by_chroma(chroma_directory=chroma_directory, model_name=model_name)

    docs = docsearch.similarity_search(query)
    return docs

def create_index_by_chroma(documents: List[Document], model_name: str = "davinci", 
                           chroma_directory: str = CHROMA_PATH) -> Chroma:
    """把文档转换为chroma的index"""
    embeddings = OpenAIEmbeddings(model=model_name)  # 必须要给出model="davinci"，否则会报错
    docsearch = Chroma.from_documents(documents, embedding=embeddings, persist_directory=chroma_directory)
    docsearch.persist()
    return docsearch

def read_index_by_chroma(chroma_directory: str = CHROMA_PATH, model_name: str = "davinci") -> Chroma:
    """从已有的index中读取文档"""
    embeddings = OpenAIEmbeddings(model=model_name)  # 必须要给出model="davinci"，否则会报错
    docsearch = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)
    return docsearch

def get_documents_from_file_path(file_path: str) -> List[Document]:
    """从文件中读取文档"""
    loader = PyPDFLoader(file_path=file_path)
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.split_documents(loader.load())
    return documents

def get_documents_from_bytes(bytes_data: bytes, source: str) -> List[Document]:
    """从bytes中读取文档"""
    loader = BackPdfLoader(bytes_data=bytes_data)
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.split_documents(loader.load())

    # 给documents加上metadata
    for i in range(len(documents)):
        documents[i].metadata["source"] = source
    return documents

