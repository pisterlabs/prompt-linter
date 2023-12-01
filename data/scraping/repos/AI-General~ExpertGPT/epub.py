from langchain.document_loaders.epub import UnstructuredEPubLoader
from models.files import File

from .common import process_file


def process_epub(file: File, enable_summarization, brain_id):
    return process_file(
        file=file,
        loader_class=UnstructuredEPubLoader,
        enable_summarization=enable_summarization,
        brain_id=brain_id
    )
