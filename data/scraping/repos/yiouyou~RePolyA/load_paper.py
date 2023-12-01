from langchain.document_loaders import ArxivLoader
from langchain.document_loaders import PubMedLoader
from langchain.document_loaders import BibtexLoader
from langchain.document_loaders import WikipediaLoader


##### ArxivLoader
def load_arxiv_to_docs(query: str):
    loader = ArxivLoader(
        query=query,
        load_max_docs=3,
    )
    docs = loader.load()
    return docs


##### PubMedLoader
def load_pubmed_to_docs(query: str):
    loader = PubMedLoader(
        query=query,
        load_max_docs=3,
    )
    docs = loader.load()
    return docs


##### BibtexLoader
def load_bibtex_to_docs(file_path: str):
    loader = BibtexLoader(
        file_path=file_path,
        max_docs=-1,
    )
    docs = loader.load()
    return docs


##### WikipediaLoader
def load_wikipedia_to_docs(query: str, lang: str = "en"):
    loader = WikipediaLoader(
        query=query,
        load_max_docs=3,
        lang=lang,
    )
    docs = loader.load()
    return docs

