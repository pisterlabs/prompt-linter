from datetime import datetime
from torch import load, save
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
import pinecone
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
import os
from pydantic import BaseModel
from typing import List, Union, Optional


def extract_data_from_urls(urls: List[str]) -> List[Document]:
    """Extract the data from a list of URLs."""
    try:
        loaders = WebBaseLoader(urls, continue_on_failure=True)
        data = loaders.load()
        return data
    except Exception as e:
        print("Could not load the data from the URLs because : " + str(e))


def split_data(data: List[Document], chunk_size=1024, chunk_overlap=256) -> List[Document]:
    """Split the data into sentences."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        text_chunks = text_splitter.split_documents(data)

        return text_chunks
    except Exception as e:
        print("Could not split the data into sentences because : " + str(e))


def saving_in_vectorstore(data: List[Document], index_name: str | None = None, store="FAISS", embeddings_type="OPENAI"):
    """Saving the data in the vectorstore."""
    try:
        if embeddings_type == "OPENAI":
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings()

        if store == "FAISS":
            vectorstore = FAISS.from_documents(data, embedding=embeddings)

            return vectorstore
        elif store == "PINECONE":
            pinecone_api_key = os.environ.get('PINECONE_API_KEY')
            pinecone_api_env = os.environ.get('PINECONE_API_ENV')

            try:
                pinecone.init(api_key=pinecone_api_key, environment=pinecone_api_env)
            except Exception as e:
                print("Could not init Pinecone because : " + str(e))

            try:
                # vectorstore = Pinecone.from_texts([t.page_content for t in data], embeddings, index_name=index_name)
                vectorstore = Pinecone.from_documents(data, embeddings, index_name=index_name)

                return vectorstore
            except Exception as e:
                print("Could not create the vectorstore because : " + str(e))
        else:
            raise Exception("The vectorstore is not valid.")

    except Exception as e:
        print("Could not save the data in the vectorstore because : " + str(e))


def creating_chain(vectorstore: Union[FAISS, Pinecone]) -> RetrievalQA:
    """Creating QA Chain."""
    try:

        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        {context}
        Question: {question}
        Helpful Answer:"""

        qa_chain_prompt = PromptTemplate.from_template(template)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt},
        )

        return qa_chain

    except Exception as e:
        print("Error while creating QA Chain : " + str(e))


class Model:
    def __init__(self, urls: List[str] | None):
        self.urls = urls
        self.vectorstore: FAISS | Pinecone | None = None
        self.chain = None
        self.creation_date = None

    def answer(self, query: str) -> str:
        """Answering the question."""

        try:
            result = self.chain({"query": query}, return_only_outputs=True)
            result = result["result"]

            return result
        except Exception as e:
            print("Error while answering the question : " + str(e))

    def save(self):
        """Saving the model."""
        try:
            # Check if there's a previous latest model, and rename it
            if os.path.exists("../models/latest.pt"):
                temp_model = load("../models/latest.pt")
                os.rename("../models/latest.pt", f"models/model_{temp_model.creation_date}.pt")

            save(self, f"../models/latest.pt")
        except Exception as e:
            print("Error while saving the model : " + str(e))

    def train(self, urls: List[str] | None, store="FAISS"):
        """Train new Q/A chatbot with data from those urls"""

        if os.environ['OPENAI_API_KEY'] is None:
            raise ValueError("OPENAI_API_KEY must be provided.")
        elif store != "FAISS" and os.environ['PINECONE_API_KEY'] is None:
            raise ValueError("PINECONE_API_KEY must be provided.")
        else:
            try:
                if self.urls is None:
                    self.urls = urls

                if self.urls is None and urls is None:
                    raise ValueError("You must provide a list of URLs.")
                else:
                    data = extract_data_from_urls(self.urls)
                    data = split_data(data)
                    if store == "FAISS":
                        self.vectorstore = saving_in_vectorstore(data)
                    else:
                        self.vectorstore = saving_in_vectorstore(data, index_name="chatbot", store=store)

                    self.chain = creating_chain(self.vectorstore)
                    self.creation_date = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

                    self.save()

            except Exception as e:
                print("Error while training the model : " + str(e))


def load_model(model_path="latest.pt"):
    try:
        model = load(f"../models/{model_path}")

        return model
    except Exception as e:
        print("Error while loading the model :" + str(e))
