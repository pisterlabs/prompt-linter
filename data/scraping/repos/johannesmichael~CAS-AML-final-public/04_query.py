from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
import argparse

load_dotenv()

from scripts.constants import CHROMA_SETTINGS


persist_directory = os.environ.get('PERSIST_DIRECTORY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

embeddings = OpenAIEmbeddings()



# create the top-level parser
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--collection_name", help="Name of the collection to create/use")
    parser.add_argument("--max_tokens", help="max tokens to generate", default=1000)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    collection_name = args.collection_name
    max_tokens = args.max_tokens

    db = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    
    retriever = db.as_retriever()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", max_tokens=max_tokens)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever )

    while True:
        query = input("> ")
        answer = qa.run(query)
        print(answer)


#python 04_query.py --collection_name openai_ada_1000cs --max_tokens 1000