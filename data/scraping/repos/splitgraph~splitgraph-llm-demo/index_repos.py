# from: https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pgvector.html

from io import StringIO
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# from: https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/markdown.html
from langchain.document_loaders import UnstructuredFileIOLoader
import langchain.vectorstores.pgvector
from unstructured.__version__ import __version__ as __unstructured_version__  # type: ignore
from unstructured.partition.md import partition_md  # type: ignore
import os
from langchain_demo.repo_info import get_repo_list
from langchain_demo.repo_to_md import repository_info_to_markdown
import sys


# Based on UnstructuredMarkdownLoader
class UnstructuredMarkdownIOLoader(UnstructuredFileIOLoader):
    """Loader that uses unstructured to load markdown files."""

    def _get_elements(self) -> List:
        _unstructured_version = __unstructured_version__.split("-")[0]
        unstructured_version = tuple([int(x) for x in _unstructured_version.split(".")])

        if unstructured_version < (0, 4, 16):
            raise ValueError(
                f"You are on unstructured version {__unstructured_version__}. "
                "Partitioning markdown files is only supported in unstructured>=0.4.16."
            )

        return partition_md(file=self.file, **self.unstructured_kwargs)


def save_embedding(collection_name, namespace, repository, md):
    loader = UnstructuredMarkdownIOLoader(StringIO(md))
    documents = loader.load()
    # add metadata to documents
    for d in documents:
        d.metadata["namespace"] = namespace
        d.metadata["repository"] = repository
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    langchain.vectorstores.pgvector.PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=collection_name,
        connection_string=os.environ["PG_CONN_STR_LOCAL"],
    )
    print(
        f"saved embeddings for {len(docs)} document fragments describing repo {namespace}/{repository}"
    )


def main():
    repo_index_limit = (
        None  # set repo_index_limit to an integer to only index the first N repos
    )
    collection_name = "repo_embeddings"
    namespace = sys.argv[1]
    print(f"Indexing repositories in namespace {namespace}")
    repo_list = get_repo_list(namespace)
    for repo_info in repo_list[0:repo_index_limit]:
        save_embedding(
            collection_name,
            repo_info["namespace"],
            repo_info["repository"],
            repository_info_to_markdown(repo_info),
        )


main()
