from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from .base_tool import BaseTool

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_transformers import DoctranQATransformer
from langchain.schema import Document
from langchain.vectorstores import LanceDB

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import lancedb
import pickle
import tempfile


class ShopperTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Shopper",
            model="gpt-4",
            temperature=0.0,
            uploads=[
                {
                    "input_label": "Upload PDF",
                    "help_label": "Upload a PDF to be used as the source document.",
                },
            ],
            inputs=[
                {
                    "input_label": "Question",
                    "example": "What is the minimum budget requirements to run a Pinterest ad?",
                    "button_label": "Ask",
                    "help_label": "The Q&A tool helps by answering a given question based on the PDF you provided.",
                },
            ],
        )

    def execute(self, chat, inputs, uploads):
        self._ingest_pdf(uploads)
        basic_qa_chain = self._basic_qa_chain(chat)
        question_input = {
            "question": inputs,
            "chat_history": [],
        }
        result = basic_qa_chain.run(question_input)
        return result

    # 1 Ingest, split, and embed PDF Docs
    def _ingest_pdf(self, uploads):
        print("Loading data...")
        uploaded_file = uploads  # Directly using file_inputs as an UploadedFile object
        file_bytes = (
            uploaded_file.read()
        )  # Reading the content of the uploaded file as bytes

        # Creating a temporary file to write the bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        loader = PDFPlumberLoader(temp_file_path)
        raw_documents = loader.load()

        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_documents = text_splitter.split_documents(raw_documents)

        print("Creating vectorstore...")
        embeddings = OpenAIEmbeddings()
        db = lancedb.connect("/tmp/lancedb")
        table = db.create_table(
            "my_table",
            data=[
                {
                    "vector": embeddings.embed_query("Hello World"),
                    "text": "Hello World",
                    "id": "1",
                }
            ],
            mode="overwrite",
        )
        vectorstore = LanceDB.from_documents(
            split_documents, embeddings, connection=table
        )
        with open("./vector_db/vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

    def _load_retriever(self):
        with open("./vector_db/vectorstore.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        retriever = VectorStoreRetriever(vectorstore=vectorstore)
        return retriever

    def _basic_qa_chain(self, chat):
        # llm = ChatOpenAI(verbose=True, model_name="gpt-4", temperature=0)
        retriever = self._load_retriever()
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=chat, retriever=retriever, memory=memory, verbose=True
        )
        return chain
