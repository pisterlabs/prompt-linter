from langchain.document_loaders import DirectoryLoader, TextLoader, PythonLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter, PythonCodeTextSplitter
from langchain.schema import Document

from ml_playground.langchain.utils.db import from_documents
from ml_playground.utils.io import LIB_DIR


def load_md_docs() -> list[Document]:
    loader = DirectoryLoader(
        f"{LIB_DIR}/langchain/data/kolena/docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = text_splitter.split_text(" ".join([d.page_content for d in documents]))
    return splits


def load_source_code() -> list[Document]:
    def _load_code(dir: str) -> list[Document]:
        loader = DirectoryLoader(
            dir,
            glob="**/*.py",
            loader_cls=PythonLoader,
            show_progress=True,
        )
        documents = loader.load()
        text_splitter = PythonCodeTextSplitter()
        splits = text_splitter.split_text(" ".join([d.page_content for d in documents]))
        return [Document(page_content=s) for s in splits]

    target_dirs = [
        f"{LIB_DIR}/langchain/data/kolena/kolena",
        f"{LIB_DIR}/langchain/data/kolena/tests",
        f"{LIB_DIR}/langchain/data/kolena/examples",
    ]
    splits = []
    for dir in target_dirs:
        splits.extend(_load_code(dir))
    return splits


def main():
    md_docs_splits = load_md_docs()
    source_code_splits = load_source_code()
    splits = md_docs_splits + source_code_splits

    embeddings = OpenAIEmbeddings()
    vectordb = from_documents(
        splits,
        embeddings,
        reset=True,
    )
    return vectordb


if __name__ == "__main__":
    main()
