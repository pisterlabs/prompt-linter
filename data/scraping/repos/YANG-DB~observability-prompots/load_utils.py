from typing import Any, List
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

# loads the documents and prepares them for embedding
from json_loader import JSONLoader
from web_loder import WebBaseLoader


def loadDocuments() -> List[Document]:
    # load opensearch observability website
    # loader = WebBaseLoader("https://opensearch.org/docs/latest/observing-your-data/index/",depth=1)
    # documents = loader.load()

    # load a documentation folder for integration
    # loader = DirectoryLoader('spec', glob="**/*.md")
    documents = []

    # load all documentation from opensearch docs/blogs
    loader = DirectoryLoader("../static_data", glob="**/*.md")
    for doc in loader.load():
        documents.append(doc)

    # load ppl samples queries
    loader = DirectoryLoader("../prompts", glob="**/*.txt")
    for doc in loader.load():
        documents.append(doc)

    # load ppl samples queries
    loader = DirectoryLoader("../samples", glob="**/*.txt")
    for doc in loader.load():
        documents.append(doc)

    # load O/S queries responses
    loader = DirectoryLoader("../queries", glob="**/*.txt")
    for doc in loader.load():
        documents.append(doc)

    # split content
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)

