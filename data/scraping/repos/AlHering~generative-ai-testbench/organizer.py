# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:organizer                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from pandas import Series
from sklearn.cluster import KMeans, DBSCAN
from langchain.prompts import PromptTemplate
from src.librarian.librarian import Librarian


class Organizer(Librarian):
    """
    Class, representing an LLM-based organizer agent to collect, embed, cluster and organize texts or documents.
    """

    def __init__(self, profile: dict) -> None:
        """
        Initiation method.
        :param profile: Profile, configuring a organizer agent. The profile should be a nested dictionary of the form
            'chromadb_settings': ChromaDB Settings.
            'embedding':
                'embedding_model': Embedding model.
                'embedding_function': Embedding function.
            'retrieval': 
                'source_chunks': Source chunks.
        """
        super().__init__(profile)
        self.prompt_template = PromptTemplate(
            template="""
                You are a classification AI. You will recieve a list of documents. The documents are delimited by ####. 
                {task}

                DOCUMENTS:
                {document_list}

                YOUR RESPONSE:
            """,
            input_variables=["task", "document_list"]
        )

    # Override
    def enrich_content_batches(self, file_paths: List[str], content_batches: List[list]) -> Tuple[list, list]:
        """
        Method for creating metadata for file contents.
        :param file_paths: File paths.
        :param content_batches: Content batches.
        :return: Metadata-enriched contents.
        """
        contents = []
        metadata_entries = []
        for file_index, document_contents in enumerate(content_batches):
            for part_index, document_part in document_contents:
                contents.append(document_part)
                metadata_entries.append({"file_path": file_paths[file_index],
                                         "part": part_index,
                                         "raw": document_part})
        return contents, metadata_entries

    def run_clustering(self, method: str = "kmeans", method_kwargs: dict = None) -> pd.DataFrame:
        """
        Method for running clustering.
        :param method: Method/Algorithm for running clustering. Defaults to 'kmeans'.
        :param method_kwargs: Keyword arguments (usually hyperparameters) for running the clustering method.
        :return: Pandas DataFrame, containing text under column "document" and the cluster label under "cluster".
        """
        embedded_docs = self.vector_db.get(include=["embeddings", "metadatas"])
        embedded_docs_df = pd.DataFrame(
            Series(entry) for entry in embedded_docs["embeddings"])

        clusters = {"kmeans": KMeans, "dbscan": DBSCAN}[
            method](**method_kwargs).fit(embedded_docs_df).labels_

        return pd.DataFrame(zip([entry["raw"] for entry in embedded_docs["metadatas"]], clusters), columns=["document", "cluster"])

    def choose_topics(self, clustered_df: pd.DataFrame) -> dict:
        """
        Method for choosing topics based on clustered texts or documents.
        :param clustered_df: DataFrame, containing clustering results.
        :return: Dictionary, mapping topics under cluster IDs.
        """
        return self.handle_clusters(clustered_df, "Your task is to find a single common topic which descibes all documents. The topic should be a single sentence.")

    def summarize_topics(self, clustered_df: pd.DataFrame) -> dict:
        """
        Method for summarizing clustered texts or documents.
        :param clustered_df: DataFrame, containing clustering results.
        :return: Dictionary, mapping summaries under cluster IDs.
        """
        return self.handle_clusters(clustered_df, "Write a summary for the documents.")

    def handle_clustered_documents(self, clustered_df: pd.DataFrame, task: str) -> dict:
        """
        Method for issueing a task on clustered documents.
        :param clustered_df: DataFrame, containing clustering results.
        :param task: Task description relative to a list of documents as an english text.
            Example: 'Your task is to choose and summarize the most funny document.'
        :return: Dictionary containing the task result for each cluster und the cluster ID as key.
        """
        result = {}
        for cluster_id in clustered_df["cluster"].unique():
            docs = list(
                clustered_df.iloc[clustered_df.cluster == cluster_id]["document"])
            prompt = self.prompt_template(
                task=task,
                document_list=" #### ".join(docs)
            )
            result[cluster_id] = self.llm(prompt)
        return result
