import glob
import os
import openai
import pinecone

from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
pinecone_api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
pinecone_env = os.getenv('PINECONE_ENVIRONMENT') or "YOUR_ENV"

def query_docs(query):
    embed_model = "text-embedding-ada-002"

    chat = ChatOpenAI(
        openai_api_key=openai.api_key,
        model="gpt-4"
    )

    embed = OpenAIEmbeddings(
        model=embed_model,
        openai_api_key=openai.api_key
    )

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index = pinecone.Index('zkappumstad')

    vector_store = Pinecone(index, embed.embed_query, "text")

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    xq = openai.Embedding.create(input=query, engine=embed_model)['data'][0]['embedding']
    res = index.query([xq], top_k = 3, include_values=True, include_metadata=True)

    return qa.run(query)

result = query_docs("How many fields elements can there be in a struct ?")
print(result)