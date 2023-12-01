import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from xata.client import XataClient


class SearchUploadedDocs:
    async def search_uploaded_docs(self, query: str, top_k: int = 16) -> list[Document]:
        """Fetch uploaded docs in similarity search."""
        username = st.session_state["username"]
        session_id = st.session_state["selected_chat_id"]
        embeddings = OpenAIEmbeddings()
        query_vector = embeddings.embed_query(query)
        results = (
            XataClient()
            .data()
            .vector_search(
                "tiangong_chunks",
                {
                    "queryVector": query_vector,  # array of floats
                    "column": "embedding",  # column name,
                    "similarityFunction": "cosineSimilarity",  # space function
                    "size": top_k,  # number of results to return
                    "filter": {
                        "username": username,
                        "sessionId": session_id,
                    },  # filter expression
                },
            )
        )

        docs = []
        for record in results["records"]:
            page_content = record["content"]
            metadata = {
                "source": record["source"],
            }
            doc = Document(
                page_content=page_content,
                metadata=metadata,
            )
            docs.append(page_content)

        return docs
