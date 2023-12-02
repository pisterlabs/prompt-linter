from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone

import pinecone
import pandas as pd

from util.logger import get_logger
from features.preprocess import remove_stopwords, lemmatize

import openai
import os
from typing import List, Dict

openai.api_key = os.environ.get("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if pinecone_api_key is None:
    raise ValueError("The PINECONE_API_KEY environment variable is not set.")


class LegalSearchService:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.text_field = "text"
        self.__init_pinecone__()
        self.embed = HuggingFaceEmbeddings(
            model_name=self.config["sentence_transformer"]["model_name"]
        )
        self.__load_vectorstore__()
        self.df_processed = pd.read_csv(
            "../data/02_interim/law/law_art_abs_text.csv", delimiter="|"
        )
        self.df_original = pd.read_csv(
            "../data/01_raw/law/law_art_abs_text.csv", delimiter="|"
        )

    def __init_pinecone__(self):
        pinecone.init(
            api_key=pinecone_api_key,
            environment=self.config["vectorization"]["environment"],
        )
        index_name_law = "law"
        self.index = pinecone.Index(index_name_law)

    def __load_vectorstore__(self):
        self.vectorstore = Pinecone(
            self.index, self.embed.embed_query, text_key=self.text_field
        )

    def search_relevant_articles(self, analyzed_situation: List[Dict]) -> List[Dict]:
        for situation in analyzed_situation:
            for action in situation["aktionen"]:
                query = self.create_query(action["beschreibung"])
                documents = self.vectorstore.similarity_search(query, k=5)
                action["artikel"] = self.get_artikel(documents)
        return analyzed_situation

    def get_artikel(self, documents: List[Dict]) -> Dict:
        artikel = {}
        for document in documents:
            print(type(document))
            self.logger.debug(f"Document: {document}")
            key, original_text = self.load_original_text_with_key(
                document.page_content
            )
            self.logger.debug(f"Key: {key}, Original Text: {original_text}")
            artikel[key] = original_text
        return artikel

    def create_query(self, query_text) -> str:
        query = query_text
        self.logger.debug(f"Original query: {query}")
        query = remove_stopwords(query)
        query = lemmatize(query)
        self.logger.debug(f"Preprocessed query: {query}")
        return query

    def load_original_text_with_key(self, preprocessed_text: str) -> [str, str]:
        self.logger.debug(f"Preprocessed text: {preprocessed_text}")
        filtered_df = self.df_processed[self.df_processed["Text"] == preprocessed_text]
        key = self.get_key(filtered_df)
        original_text = self.get_original_text(filtered_df)
        return key, original_text

    def get_key(self, df: pd.DataFrame) -> str:
        merged_column = df["Gesetz"] + " " + df["Artikel"] + " " + df["Absatz"]
        key = merged_column.values[0]
        self.logger.debug(f"Key: {key}")
        return key

    def get_original_text(self, df: pd.DataFrame) -> str:
        gesetz = df["Gesetz"].iloc[0]
        artikel = df["Artikel"].iloc[0]
        absatz = df["Absatz"].iloc[0]
        df = self.df_original[
            (self.df_original["Gesetz"] == gesetz)
            & (self.df_original["Artikel"] == artikel)
            & (self.df_original["Absatz"] == absatz)
        ]
        original_text = df["Text"].iloc[0]
        self.logger.debug(f"Original text: {original_text}")
        return original_text