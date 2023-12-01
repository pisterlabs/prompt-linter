from .common import process_file
from langchain.document_loaders import TextLoader
from sqlite3 import Connection


def process_txt(file, stats_db):
    return process_file(file, TextLoader, ".txt", stats_db=stats_db)
