from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter


def load_pdf(pdf_path: str):
    """
    :param pdf_path: file path of pdf to load
    :return: a list of documents, one element for each page
    """
    loader = PyPDFLoader(pdf_path)

    # load pages from pdf
    document_pages = loader.load()
    return document_pages


def split_into_chunks(document_pages: list[Document], chunk_size=200):
    """
    :param chunk_size: chunk size
    :param document_pages: document (as a list of pages) to split
    :return: a list of documents, one element for each split
    """
    # split texts to paragraphs
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    text_split = text_splitter.split_documents(document_pages)
    return text_split


def split_document_paragraphs(document: Document) -> list[Document]:
    new_documents = []
    raw_text = document.page_content
    paragraphs = split_text_paragraphs(raw_text)

    for par in paragraphs:
        new_documents.append(
            Document(page_content=par, metadata=document.metadata))

    return new_documents


def split_documents_paragraphs(documents: list[Document]) -> list[Document]:
    result = []
    for doc in documents:
        result += split_document_paragraphs(doc)
    return result


def split_text_paragraphs(text: str):
    # Split text into paragraphs using double line breaks
    paragraphs = text.split('\n\n')
    return paragraphs
