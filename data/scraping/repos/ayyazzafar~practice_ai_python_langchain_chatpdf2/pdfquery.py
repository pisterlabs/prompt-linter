# Importing necessary modules
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


class PDFQuery:
    def __init__(self, openai_api_key=None) -> None:
        """
        Initialize the PDFQuery class.

        Args:
        openai_api_key (str): The API key for OpenAI.
        """

        # Initialize OpenAIEmbeddings with OpenAI API key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Set OpenAI API key in the environment variables
        os.environ["OPENAI_API_KEY"] = openai_api_key
        # Initialize a text splitter for splitting the document into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        # Initialize a language model
        self.llm = OpenAI(
            temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")
        # Initialize the question answering chain and database to None
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        """
        Ask a question to the system.

        Args:
        question (str): The question to ask.

        Returns:
        str: The system's response.
        """

        # If no document is loaded, prompt the user to add a document
        if self.chain is None:
            response = "Please, add a document."
        else:
            # Retrieve relevant documents for the question
            docs = self.db.get_relevant_documents(question)
            # Run the question answering chain on the relevant documents
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, file_path: os.PathLike) -> None:
        """
        Ingest a document for processing.

        Args:
        file_path (os.PathLike): The path to the document.
        """

        # Load the document using the PyPDFium2Loader
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        # Split the document into chunks
        splitted_documents = self.text_splitter.split_documents(documents)
        # Generate a retriever from the document chunks
        self.db = Chroma.from_documents(
            splitted_documents, self.embeddings).as_retriever()
        # Load the question answering chain
        self.chain = load_qa_chain(
            OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), chain_type="stuff")

    def forget(self) -> None:
        """
        Forget the current document and question answering chain.
        """
        self.db = None
        self.chain = None
