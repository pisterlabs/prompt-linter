from langchain.document_loaders import PyPDFLoader


def extract_paper_content(url):
    loader = PyPDFLoader(url)
    documents = loader.load_and_split()
    raw_text = "\n\n".join([document.page_content for document in documents])
    return raw_text
