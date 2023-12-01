from .common import process_file
from langchain.document_loaders import UnstructuredMarkdownLoader
from sqlite3 import Connection


def process_markdown(file, stats_db):
    return process_file(file, UnstructuredMarkdownLoader, ".md", stats_db=stats_db)
