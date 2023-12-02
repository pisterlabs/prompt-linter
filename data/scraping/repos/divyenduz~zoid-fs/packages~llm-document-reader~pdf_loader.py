"""
PDFLoader: Load a PDF as text or vectorstore
"""
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


class PDFLoader:
    """
    PDFLoader: Load a PDF as text or vectorstore
    """

    def __init__(self, pdf_path_or_bytes):
        if isinstance(pdf_path_or_bytes, str):
            self.loader = PyPDFLoader(pdf_path_or_bytes, extract_images=True)
        elif isinstance(pdf_path_or_bytes, bytes):
            with tempfile.NamedTemporaryFile(delete=False) as file_like:
                file_like.write(pdf_path_or_bytes)
                file_like.seek(0)
                self.loader = PyPDFLoader(file_like.name, extract_images=True)
        else:
            raise ValueError(
                "Invalid argument: pdf_path_or_bytes must be a string or bytes."
            )

    def open_pdf(self):
        """Open a PDF and return a list of pages"""
        pages = self.loader.load()
        return pages

    def vector_store(self):
        """Open a PDF and return a vector store"""
        pages = self.open_pdf()
        vectorstore = FAISS.from_documents(pages, embedding=OpenAIEmbeddings())
        return vectorstore
