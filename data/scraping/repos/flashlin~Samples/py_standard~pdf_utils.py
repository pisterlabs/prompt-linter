from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# loader = PyPDFLoader("./attention_is_all_you_need.pdf")
# pages = loader.load_and_split()

def load_pdf_documents_from_directory(directory_path):
    loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def load_txt_documents_from_directory(directory_path):
    loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents


def splitting_documents_into_texts(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts


def load_and_split_pdf_texts_from_directory(directory_path):
    pdf_documents = load_pdf_documents_from_directory(directory_path)
    txt_documents = load_txt_documents_from_directory(directory_path)
    documents = pdf_documents + txt_documents
    return splitting_documents_into_texts(documents)
