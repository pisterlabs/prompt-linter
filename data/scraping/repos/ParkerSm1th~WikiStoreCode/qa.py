"""Ask a question to the notion database."""
import argparse
import pickle

import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
parser.add_argument('--spaceId', dest='spaceId', type=str, help='The space ID to use')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index(f'faiss_store_{args.spaceId}.index')

with open(f'faiss_store_{args.spaceId}.pkl', "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
