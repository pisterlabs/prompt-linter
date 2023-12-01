import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from getpass import getpass
import textwrap
import docx

# Check if the OpenAI API key is set
if os.environ.get("OPENAI_API_KEY") is None:
    token = getpass("Enter your OpenAI token: ")
    os.environ["OPENAI_API_KEY"] = str(token)

# Initialize OpenAI embeddings and QA chain
embeddings = OpenAIEmbeddings()
chain = load_qa_chain(OpenAI(), chain_type="stuff")


class DocSearch:
    def __init__(self, root_files):
        """
        Initialize the DocSearch object.

        Parameters
        ----------
        root_files: str or list
            A file path or a list of file paths to be processed.
        """
        if isinstance(root_files, str):
            self._root_files = [root_files]
        else:
            self._root_files = root_files
        self.docsearch = self.extract_texts(self._root_files)

    def extract_texts(self, root_files):
        """
        Extract text from uploaded files and create a FAISS index.

        Supported file types: .pdf, .docx
        If multiple files are provided, their contents are concatenated.

        Parameters
        ----------
        root_files: list
            A list of file paths to be processed.

        Returns
        -------
        FAISS index object
            Contains the embeddings of the text chunks.
        """
        raw_text = ""

        if isinstance(root_files, str):
            root_files = [root_files]

        for root_file in root_files:
            _, ext = os.path.splitext(root_file)
            if ext == ".pdf":
                with open(root_file, "rb") as f:
                    reader = PdfReader(f)
                    for i in range(len(reader.pages)):
                        page = reader.pages[i]
                        raw_text += page.extract_text()
            elif ext == ".docx":
                doc = docx.Document(root_file)
                for paragraph in doc.paragraphs:
                    raw_text += paragraph.text

        # Retrieval to avoid token size limits
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        texts = text_splitter.split_text(raw_text)

        docsearch = FAISS.from_texts(texts, embeddings)
        return docsearch

    def run_query(self, query):
        """
        Run a query on a PDF file using docsearch and the chain library.

        Parameters
        ----------
        query: str
            The query to be searched.

        Returns
        -------
        str
            The output of the chain library run on the documents returned by docsearch similarity search.
        """
        docs = self.docsearch.similarity_search(query)
        return chain.run(input_documents=docs, question=query)

    def ask_question(self, query, endpoint=None):
        """
        Return the answer to a question asked on a document.

        Parameters
        ----------
        query: str
            The query to be run.
        endpoint: function, optional
            A function that takes a string and outputs it somewhere.
            If not provided, defaults to print.

        Returns
        -------
        None
        """
        if not endpoint:
            endpoint = print

        wrapped_text = textwrap.wrap(self.run_query(query), width=60)
        for line in wrapped_text:
            endpoint(line)
