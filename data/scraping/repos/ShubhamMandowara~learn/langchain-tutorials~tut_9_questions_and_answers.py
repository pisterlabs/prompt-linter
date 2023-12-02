from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstore import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from typing import List



def load_data_from_url(url:str) -> List:
    """

    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs



def main(url:str, query:str, model_name:str='gpt-3.5-turbo-16k', temperature:float=0, openai_api_key:str=None):
    docs = load_data_from_url(url)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    #  1- 100 -> 90-190

    documents = text_splitter.split_documents(docs)

    db = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

    llm = ChatOpenAI(temperature=temperature, model_name=model_name, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    result = qa_chain({'query': query})
    return result
