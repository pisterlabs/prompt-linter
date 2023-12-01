import asyncio
import datetime

import pinecone
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


class SearchPinecone:
    def __init__(self):
        self.pinecone_api_key = st.secrets["pinecone_api_key"]
        self.pinecone_environment = st.secrets["pinecone_environment"]
        self.pinecone_index = st.secrets["pinecone_index"]
        self.openai_api_key = st.secrets["openai_api_key"]

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)

        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment,
        )
        # self.vectorstore = Pinecone.from_existing_index(
        #     index_name=self.pinecone_index,
        #     embedding=self.embeddings,
        # )

    async def async_similarity(
        self, query: str, filters: dict = {}, top_k: int = 16, extend: int = 0
    ):
        """Search Pinecone."""
        if top_k == 0:
            return []

        index = pinecone.Index(self.pinecone_index)
        vector = self.embeddings.embed_query(query)
        query_response = index.query(
            top_k=top_k,
            include_values=False,
            include_metadata=True,
            vector=vector,
            filter=filters,
        )
        docs = [
            {
                "id": match["id"],
                "metadata": match["metadata"],
            }
            for match in query_response["matches"]
        ]

        if extend > 0:
            for doc in docs:
                doc_id = doc["id"]
                id_prefix, id_suffix = doc_id.rsplit("_", 1)
                id_suffix = int(id_suffix)

                ids = [
                    f"{id_prefix}_{i}"
                    for i in range(max(0, id_suffix - extend), id_suffix + extend + 1)
                    if i != id_suffix
                ]

                response = index.fetch(ids=ids)
                adjacent_docs = response["vectors"]
                processed_adjacent_docs = [
                    {"id": key, "metadata": {"text": value["metadata"].get("text")}}
                    for key, value in adjacent_docs.items()
                ]
                processed_adjacent_docs.append(doc)

                sorted_docs = sorted(
                    processed_adjacent_docs, key=lambda x: int(x["id"].split("_")[-1])
                )
                texts = [item["metadata"]["text"] for item in sorted_docs]
                doc["metadata"]["text"] = " ".join(texts)

        docs_list = []
        for doc in docs:
            date = datetime.datetime.fromtimestamp(doc["metadata"]["created_at"])
            formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
            source_entry = "[{}. {}. {}. {}.]({})".format(
                doc["metadata"]["source_id"],
                doc["metadata"]["source"],
                doc["metadata"]["author"],
                formatted_date,
                doc["metadata"]["url"],
            )
            docs_list.append(
                {"content": doc["metadata"]["text"], "source": source_entry}
            )

        return docs_list

    def sync_similarity(
        self, query: str, filters: dict = {}, top_k: int = 16, extend: int = 0
    ):
        return asyncio.run(self.async_similarity(query, filters, top_k, extend))

    async def async_mmr(self, query: str, filters: dict = {}, top_k: int = 16):
        """Search Pinecone with maximal marginal relevance method."""
        if top_k == 0:
            return []

        if filters:
            docs = self.vectorstore.max_marginal_relevance_search(
                query, k=top_k, filter=filters
            )
        else:
            docs = self.vectorstore.max_marginal_relevance_search(query, k=top_k)

        docs_list = []
        for doc in docs:
            date = datetime.datetime.fromtimestamp(doc.metadata["created_at"])
            formatted_date = date.strftime("%Y-%m")  # Format date as 'YYYY-MM'
            source_entry = "[{}. {}. {}. {}.]({})".format(
                doc.metadata["source_id"],
                doc.metadata["source"],
                doc.metadata["author"],
                formatted_date,
                doc.metadata["url"],
            )
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list

    def sync_mmr(self, query: str, filters: dict = {}, top_k: int = 16):
        return asyncio.run(self.async_mmr(query, filters, top_k))

    def get_contentslist(self, docs):
        """Get a list of contents from docs."""
        contents = [[item["content"] for item in sublist] for sublist in docs]
        return contents
