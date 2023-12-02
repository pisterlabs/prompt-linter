from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from pydantic import BaseModel
from build_vector_database import build_and_get_database

from src.config.config import (
    OPEN_AI_LLM_MODEL,
    OPENAI_API_KEY,
)

llm = ChatOpenAI(
    temperature=0.0, model=OPEN_AI_LLM_MODEL, openai_api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = build_and_get_database()


PERFUME_TOOL_DESCRIPTION = """Useful for finding a perfume by name, brand, or fragrance notes.
    Can filter perfume by price metadata.
    Information in database is in Russian.
    Action: search for a perfume by name, brand, or fragrance notes.
    Action Input: name, brand, or fragrance notes
    Action Output: perfume data with url, name and price.
    """

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="Source URL for the perfume",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="name",
        description="Name of the perfume",
        type="string",
    ),
    AttributeInfo(
        name="price",
        description="The price of the perfume",
        type="number",
    ),
]


class PerfumeSearchTool(BaseModel):
    name: str = "perfume_search"
    description: str = PERFUME_TOOL_DESCRIPTION

    @staticmethod
    def run(input: str):
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_contents="Description of the perfume",
            metadata_field_info=metadata_field_info,
            verbose=True,
        )

        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        result = retrieval_qa({"query": input})
        answer = result["result"]
        docs = result["source_documents"]
        answer = answer + "\n---"
        for doc in docs:
            answer = (
                answer
                + "\nName: "
                + doc.metadata["name"]
                + ", URL: "
                + doc.metadata["source"]
                + ", Price:"
                + str(doc.metadata["price"])
            )

        return answer
