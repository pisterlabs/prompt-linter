from datetime import datetime
from pathlib import Path
from llama_index import download_loader, Document
import uuid
from langchain.document_loaders import UnstructuredWordDocumentLoader

PyMuPDFReader = download_loader("PyMuPDFReader")

PyMuPDFReader = download_loader("PyMuPDFReader")
# DocxReader = download_loader("DocxReader")
DocxReader = UnstructuredWordDocumentLoader
PAGE_SEP = "\n<page-break>\n"


def load_pdf(filepath, extra_info=None):
    now = datetime.now().strftime("%Y-%m-%d ## %H:%M:%S")
    filename = filepath.split("/")[-1]
    loader = PyMuPDFReader()
    documents = loader.load(file_path=Path(filepath), metadata=False)
    doc_id = documents[0].doc_id
    text = PAGE_SEP.join([i.text.decode("utf-8") for i in documents])
    if type(extra_info) == dict:
        extra_info.update(
            {"filename": filename, "filepath": filepath, "created_at": now}
        )
    else:
        extra_info = {"filename": filename, "filepath": filepath, "created_at": now}
    doc = Document(text=text, doc_id=doc_id, extra_info=extra_info)
    return [doc]


def load_docx(filepath, extra_info=None):
    now = datetime.now().strftime("%Y-%m-%d ## %H:%M:%S")
    filename = filepath.split("/")[-1]
    loader = DocxReader
    documents = loader(Path(filepath)).load()
    doc_id = documents[0].doc_id
    text = PAGE_SEP.join([i.page_content for i in documents])
    if type(extra_info) == dict:
        extra_info.update(
            {"filename": filename, "filepath": filepath, "created_at": now}
        )
    else:
        extra_info = {"filename": filename, "filepath": filepath, "created_at": now}
    doc = Document(text=text, doc_id=doc_id, extra_info=extra_info)
    return [doc]


class IngestDocument:
    """
    Ingest all types of documents return: doc_id, text, filename, filepath, title, summary, timeofcreation,  if can be inferred
    """

    allowed = ["pdf", "docx"]

    def __init__(self):
        pass

    def load_document(self, filepath, **kwargs):
        ext = filepath.split(".")[-1]

        assert (type(ext) == str) and (len(ext) > 0), "invalid extension"
        if ext in self.allowed:
            document = self.process_allowed_types(filepath, ext, **kwargs)
            return document

    def process_allowed_types(self, filename, ext, **kwargs):
        if ext == "pdf":
            return load_pdf(filename, **kwargs)
        elif ext == "docx":
            return load_docx(filename, **kwargs)
        else:
            raise TypeError(f"Invalid type {ext} : file {filename}")
