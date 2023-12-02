from langchain.document_loaders import TextLoader
from langchain.text_splitter import text_splitter


def get_paper(paper: str) -> str:
    """
    Queries SciHub database and returns a text file with a paper.
    """
    loader = TextLoader("https://sci-hub.se/" + paper)
    document = loader.load()
    text = text_splitter(document)
    return document
