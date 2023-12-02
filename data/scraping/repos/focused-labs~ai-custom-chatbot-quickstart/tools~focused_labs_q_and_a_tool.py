import os

import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from config import PINECONE_ENVIRONMENT, PINECONE_INDEX, EMBEDDING_MODEL

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def create_vector_db_tool(llm: ChatOpenAI):
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=PINECONE_ENVIRONMENT
    )
    text_field = "text"

    index = pinecone.Index(PINECONE_INDEX)

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = Pinecone(
        index, embedding_model.embed_query, text_field
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        input_key="question",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
