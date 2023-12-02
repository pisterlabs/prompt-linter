from langchain.document_loaders import  PyPDFDirectoryLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator


loader = PyPDFDirectoryLoader(path="docs")


def make_index_from_docs():
    index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
    ).from_loaders([loader])
    return index


def query_index_citation(index, query):
    response = index.query_with_sources(query)
    return response