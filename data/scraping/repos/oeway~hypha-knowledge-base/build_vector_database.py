import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


SCIKIT_IMAGE_VERSION = os.environ.get("SCIKIT_IMAGE_VERSION")
assert SCIKIT_IMAGE_VERSION is not None, "Please set SCIKIT_IMAGE_VERSION environment variable"

def create_embeddings_from_pdf(pdf_path, index_name, output_dir):
    # convert pdf to texts
    loader = PyPDFLoader(pdf_path)
    # split texts into chunks
    doc = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(doc)
    # create embeddings
    embeddings = OpenAIEmbeddings()
    print(f"Creating embeddings (#documents={len(documents)}))")
    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(output_dir, index_name=index_name)
    return vectordb

if __name__ == "__main__":
    vectordb = create_embeddings_from_pdf(f"docs/scikit-image-{SCIKIT_IMAGE_VERSION}.pdf", index_name="scikit-image", output_dir="docs/vectordb")
    print("Embeddings created")

    vectordb = FAISS.load_local(folder_path="docs/vectordb", index_name="scikit-image", embeddings=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(score_threshold=0.4)
    items = retriever.get_relevant_documents("scikit-image release", verbose=True)
    print(items)
    
