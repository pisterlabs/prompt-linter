from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document
from typing import List
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Genie:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = TextLoader(self.file_path)
        self.documents = self.loader.load()
        self.texts = self.text_split(self.documents)
        self.vectordb = self.embeddings(self.texts)
        self.retriever = self.vectordb.as_retriever()
        self.genie = ConversationalRetrievalChain.from_llm(
                        llm=OpenAI(temperature=0), 
                        chain_type="stuff", 
                        retriever=self.retriever,
                        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                    )

    @staticmethod
    def text_split(documents: TextLoader):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    @staticmethod
    def embeddings(texts: List[Document]):
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(texts, embeddings)
        return vectordb

    def ask(self, query: str):
        return self.genie({"question": query})["answer"]

if __name__ == "__main__":
    genie = Genie("text_files/sample.txt")
    while True:
        user_input = input("Enter a query: ")
        if user_input == "exit":
            break

        try:
            print(genie.ask(user_input))
        except Exception as err:
            print('Exception occurred. Please try again', str(err))