from langchain.llms import GooglePalm
import google.generativeai as palm
from langchain.chat_models import ChatGooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Any, List, Optional
from langchain.document_loaders import (Docx2txtLoader, JSONLoader,
                                        PyPDFLoader, TextLoader,
                                        UnstructuredEmailLoader,
                                        UnstructuredHTMLLoader)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain




palm.configure(api_key="AIzaSyAm964k0ZGic7mo1Dbj6VUKqDVorfQ9xKc")

pdf_path1 = '/Users/sahibsingh05/Documents/code/py/Employee_handbook.pdf'


class CustomException(Exception):
    """Base class for other exceptions"""
    pass


class FileNameError(CustomException):
    """Raise when the file name provided does not match the given requirements"""
    pass


class LoadDocument():
    def __init__(self,
                 jq_schema: Optional[str] = None,
                 content_key:  Optional[str] = None,
                 metadata_fields:  Optional[List[str]] = None,
                 chunk_size: Optional[int] = 100,
                 chunk_overlap: Optional[int] = 0,
                 user_metadata: Optional[dict] = {},
                 file_id: Optional[str] = None,
                 data: Optional[str] = None):

        self.path = data
        self.jq_schema = jq_schema
        self.content_key = content_key
        self.metadata_fields = metadata_fields
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.user_metadata = user_metadata
        self.file_id = file_id
        self.data = data

    def classify_document(self):
        # classify data into file format and use respective loader
  
        if self.path.endswith('txt'):
            self.data = self.load_text()
        else:
            raise FileNameError("File Type Exception: File Type Not Supported")

    def split_data(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_documents(self.data)
        self.texts = texts
        return texts

    def split_texts(self):
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        texts = text_splitter.split_text(self.data)
        documents = text_splitter.create_documents(texts)
        texts = documents
        new_texts = []
        for text in texts:
            text.metadata.update(self.user_metadata)
            if self.file_id:
                text.metadata.update({"file_id": self.file_id})
            new_texts.append(text)
        texts = new_texts
        self.texts = texts
        return texts

    def load_text(self):
        loader = TextLoader(self.path)
        data = loader.load()
        return data

def data_gen(txt_path):
    pdf_obj = LoadDocument(data=txt_path, chunk_size=3000, chunk_overlap=200)
    pdf_obj.classify_document()
    docs = pdf_obj.split_data()
    embeddings = GooglePalmEmbeddings(
        google_api_key="AIzaSyAm964k0ZGic7mo1Dbj6VUKqDVorfQ9xKc")
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    return vector_store



def run_knowledge_llm(user_inpt):
    from langchain.prompts import PromptTemplate
    # input document and embedding fuction to convert the docs into vector embeddings

    combine_docs_prompt = PromptTemplate.from_template('')
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    llm = GooglePalm(google_api_key='AIzaSyAm964k0ZGic7mo1Dbj6VUKqDVorfQ9xKc')
    qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=data_gen(
        pdf_path1).as_retriever(k=2, fetch_k=2), combine_docs_chain_kwargs=dict(prompt=combine_docs_prompt))
    query = user_inpt

    answer = qa(query)
    print(answer['answer'])