from typing import Optional

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

from constants import QDRANT_PATH
from models.api import Collection

load_dotenv(verbose=True)

SYSTEM_PROMPT = """
質問に回答してください。
もし答えがわからない場合は、わからないと回答してください。
"""


def answer(
    question: str,
    collection: Collection | None = None,
    temperature: Optional[float] = None,
) -> str:
    if temperature is None:
        temperature = 0

    if collection is None:
        chat_model = ChatOpenAI()

        return chat_model.predict(question)

    embeddings = OpenAIEmbeddings()
    client = QdrantClient(path=QDRANT_PATH)
    qdrant = Qdrant(client, collection_name=collection, embeddings=embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=qdrant.as_retriever()
    )
    return qa.run(question)
