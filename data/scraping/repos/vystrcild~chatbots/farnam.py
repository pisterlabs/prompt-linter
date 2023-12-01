from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain import VectorDBQA
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def init_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(persist_directory="app/utilities/farnam/db", embedding_function=embeddings)
    llm = OpenAI(temperature=0.4, openai_api_key=OPENAI_API_KEY, verbose=True)
    chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=vectordb, return_source_documents=True)
    return chain

# chain = init_chain()
# result = chain("Who is Shaun Perish?")
# print(result)

def generate_farnam_reply(text):
    chain = init_chain()
    result = chain(text)
    reply = {
        "user": "Farnam Street",
        "type": "ai",
        "text": str(result["result"]).lstrip(),
        "datetime": str(datetime.now()),
        "room": "farnam"
    }
    return reply
