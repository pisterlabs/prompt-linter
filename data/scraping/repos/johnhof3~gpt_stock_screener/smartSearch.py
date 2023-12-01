# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pinecone.html used as reference

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import pinecone
import os
import openai

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(client=OPENAI_API_KEY)

def semantic_search(query):
    if PINECONE_API_KEY is not None and PINECONE_API_ENV is not None and PINECONE_INDEX_NAME is not None:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        docsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
        relevent_data = docsearch.similarity_search(query)

        llm = OpenAI(client=OPENAI_API_KEY)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        print(chain.run(input_documents=relevent_data, question=query))

semantic_search("What are the primary costs of revenue?")