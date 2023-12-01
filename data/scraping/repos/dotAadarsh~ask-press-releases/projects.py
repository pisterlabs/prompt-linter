import json
import os
import uuid
from typing import Dict, List

import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
import streamlit as st

st.set_page_config(page_title="Semantic Press", page_icon='🗞️', layout="centered", initial_sidebar_state="auto", menu_items=None)



QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_HOST = st.secrets["QDRANT_HOST"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

COHERE_SIZE_VECTOR = 4096  # Larger model

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is not set")

if not QDRANT_HOST:
    raise ValueError("QDRANT_HOST is not set")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is not set")


class SearchClient:
    def __init__(
        self,
        qdrabt_api_key: str = QDRANT_API_KEY,
        qdrant_host: str = QDRANT_HOST,
        cohere_api_key: str = COHERE_API_KEY,
        collection_name: str = "pressReleases",
    ):
        self.qdrant_client = QdrantClient(host=qdrant_host, api_key=qdrabt_api_key)
        self.collection_name = collection_name

        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=COHERE_SIZE_VECTOR, distance=models.Distance.COSINE
            ),
        )

        self.co_client = cohere.Client(api_key=cohere_api_key)

    # Qdrant requires data in float format
    def _float_vector(self, vector: List[float]):
        return list(map(float, vector))

    # Embedding using Cohere Embed model
    def _embed(self, text: str):
        return self.co_client.embed(texts=[text]).embeddings[0]
    
    # Prepare Qdrant Points
    def _qdrant_format(self, data: List[Dict[str, str]]):
        points = [
            models.PointStruct(
                id=uuid.uuid4().hex,
                payload={"title": point["title"], "contents": point["contents"]},
                vector=self._float_vector(self._embed(point["contents"])),
            )
            for point in data
        ]

        return points

    # Index data
    def index(self, data: List[Dict[str, str]]):
        """
        data: list of dict with keys: "title" and "contents"
        """

        points = self._qdrant_format(data)

        result = self.qdrant_client.upsert(
            collection_name=self.collection_name, points=points
        )

        return result

    # Search using text query
    def search(self, query_text: str, limit: int = 3):
        query_vector = self._embed(query_text)

        return self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=self._float_vector(query_vector),
            limit=limit,
        )

if __name__ == "__main__":

    st.header("SemanticPress")
    st.caption("Discover the power of semantic search in government press releases with ease - Powered by Qdrant x Cohere")

    with st.sidebar:
        st.info("The data is from Department of Justice 2009-2018 Press Releases - [Kaggle](https://www.kaggle.com/datasets/jbencina/department-of-justice-20092018-press-releases).")
        with st.expander("Context"):
            st.write("""This is a historical dataset containing 13,087 press releases from the Department of Justice's (DOJ) website https://www.justice.gov/news. The DOJ typically publishes several releases per day and this dataset spans from 2009 to July 2018. The releases contain information such as outcomes of criminal cases, notable actions taken against felons, or other updates about the current administration. This dataset only includes releases categorized as "Press release" and does not contain those which have been labeled as "Speeches". Some releases are tagged with topics or related agencies.""")
        with st.expander("Content"):
            st.write("""
            The contents are stored as newline delimited JSON records with the following fields:

- id: Press release number (can be missing if included in contents)
- title: Title of release
- contents: Text of release
- date: Posted date
- topics: Array of topic tags (if any provided)
- components: Array of agencies & departments (if any provided)'
The dataset has been modified and reduces to 100 titles.
""")
    
    
    client = SearchClient()


    # import data from data.json file
    with open("data.json", "r") as f:
        data = json.load(f)
    
    index_result = client.index(data)
    
    with st.sidebar:
        with st.expander("Suggested search", expanded=True):
            st.write("""
            - Racketeer Influenced and Corrupt Organizations Act (RICO)
            - health care fraud
            - Health Care Fraud Prevention & Enforcement Action Team
            """)

    query = st.text_input("Enter your query", value="public health risk")

    if query is not None:
        search_result = client.search(
            query,
        )

    i = 1
    for item in search_result:
        with st.expander(f"Press release - {i}"):
            st.write(item)
            i += 1