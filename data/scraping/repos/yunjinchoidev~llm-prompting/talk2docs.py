from typing import Any, List, Dict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import pinecone
import os
from load_dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Dict[str, Any]]) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name="news", embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )

    # 메모리
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    query = "what is langchain?"
    result = run_llm(query=query)
    print(result)
