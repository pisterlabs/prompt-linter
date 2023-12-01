from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import (
    UnstructuredMarkdownLoader,
    DirectoryLoader,
    NotebookLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


persist_directory = "db"
embedding = OpenAIEmbeddings(
    model="text-embedding-ada-002", client=None, chunk_size=4000
)


# TODO: support upserting?
def run():
    print("parsing docs...")
    md_loader = DirectoryLoader(
        ".",
        glob="docs/**/*.md",
        use_multithreading=True,
        show_progress=True,
        loader_cls=UnstructuredMarkdownLoader,
    )
    md_docs = md_loader.load()
    ipynb_loader = DirectoryLoader(
        ".",
        glob="docs/**/*.ipynb",
        use_multithreading=True,
        show_progress=True,
        loader_cls=NotebookLoader,
        loader_kwargs={
            "include_outputs": True,
            "max_output_length": 40,
            "remove_newline": True,
        },
    )
    ipynb_docs = ipynb_loader.load()
    print("generating embeddings...")

    docs = md_docs + ipynb_docs
    docs = [doc for doc in docs if "data:image" not in doc.lc_kwargs["page_content"]]
    ids = [doc.metadata["source"] for doc in docs]

    chunk_size = 50
    vectordb = None
    for i in range(0, len(docs), chunk_size):
        print(f"Processing from {i} to {i + chunk_size}")
        if vectordb is None:
            vectordb = Chroma.from_documents(
                collection_name="langchain",
                documents=docs[i : i + chunk_size],
                ids=ids[i : i + chunk_size],
                embedding=embedding,
                persist_directory=persist_directory,
            )
        else:
            vectordb.add_documents(
                docs[i : i + chunk_size], ids=ids[i : i + chunk_size]
            )
    vectordb.persist()  # type: ignore

    print("indexing done")


if __name__ == "__main__":
    run()
