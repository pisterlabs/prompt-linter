from langchain.document_loaders import UnstructuredPDFLoader

def pdf_to_structure(path: str):
    loader = UnstructuredPDFLoader("example_data/layout-parser-paper.pdf")
    data = loader.load()
    return data