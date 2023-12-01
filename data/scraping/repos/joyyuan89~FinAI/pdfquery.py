import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
#from langchain.document_loaders import PyPDFium2Loader
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
#from langchain.chat_models import ChatOpenAI


class PDFQuery:

    def __init__(self, openai_api_key = None) -> None:
        #self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        #os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.chain = None
        self.db = None
        #self.retriver = None

    def qa(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def upload(self, file_path: os.PathLike) -> None:
        #loader = PyPDFium2Loader(file_path)
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        chunks = pages
        #chunks = loader.load_and_split(text_splitter = self.text_splitter)
        #chunks = self.text_splitter.split_documents(pages)
        
        self.db = FAISS.from_documents(chunks, self.embeddings).as_retriever(search_type="mmr")
        self.chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    def forget(self) -> None:
        self.db = None
        self.chain = None
