"""This one creates a vectorstore from the training data, and then uses it to
find the top N most similar documents to each document in the test data.

Then, finds the closest k documents to each document in the test data.

It has two strategies for determining the predicted label:
1. Majority vote
2. pick the closest M documents, and then pick the label that appears the most
"""

import csv
from io import TextIOWrapper
from typing import Dict, List, Optional, Sequence

import numpy
import pandas
from langchain.docstore.document import Document
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from .constants import VECTOR_DB_DIRECTORY


class CustomCSVLoader(CSVLoader):
    def __init__(
        self,
        file_path: str,
        source_column: Optional[str] = None,
        metadata_columns: Sequence[str] = (),
        csv_args: Optional[Dict] = None,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
        columns_to_skip: Optional[List[str]] = None,
    ):
        super().__init__(
            file_path,
            source_column,
            metadata_columns,
            csv_args,
            encoding,
            autodetect_encoding,
        )
        self.columns_to_skip = columns_to_skip or []

    def __read_file(self, csvfile: TextIOWrapper) -> List[Document]:
        docs = []

        csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
        for i, row in enumerate(csv_reader):
            try:
                source = (
                    row[self.source_column]
                    if self.source_column is not None
                    else self.file_path
                )
            except KeyError:
                raise ValueError(
                    f"Source column '{self.source_column}' not found in CSV file."
                )

            # Skip specific columns by excluding them from the content
            content = "\n".join(
                f"{k.strip()}: {v.strip()}"
                for k, v in row.items()
                if k not in self.metadata_columns
                and k not in self.columns_to_skip
            )

            metadata = {"source": source, "row": i}
            for col in self.metadata_columns:
                try:
                    metadata[col] = row[col]
                except KeyError:
                    raise ValueError(
                        f"Metadata column '{col}' not found in CSV file."
                    )
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)

        return docs


def get_vectorstore(file_path, persist_directory=VECTOR_DB_DIRECTORY):
    loader = CustomCSVLoader(
        file_path=file_path,
        source_column="text",
        # metadata_columns=["label"],
        columns_to_skip=["Unnamed: 0"],
    )
    docs = loader.load()

    vector_db = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(),
        # persist_directory=persist_directory,
    )

    # vectordb._persist_directory = persist_directory
    return vector_db


def disease_finder_v2():
    vector_db = get_vectorstore("data/train.csv")
    df = pandas.read_csv("data/test.csv")

    k = 5
    # top_n = 5
    cnts = numpy.zeros(k)

    oe_embedder = OpenAIEmbeddings()
    for text, label in zip(df.text, df.label):
        embedding_vector = oe_embedder.embed_query(text)
        top_n_results = vector_db.similarity_search_by_vector(
            embedding_vector, k=k
        )

        # top_n_results = vector_db.max_marginal_relevance_search(
        #    text, fetch_k=top_n, k=k
        # )
        pred_labels = [doc.metadata.get("label") for doc in top_n_results]
        print(f"{label=}, {pred_labels=}")

        for i in range(k):
            cnts[i] += int(pred_labels.count(label) > i)

    for i in range(k):
        print(f"at least {i+1} match", cnts[i] / len(df))

    return cnts / len(df)
