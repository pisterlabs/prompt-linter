import argparse
import faiss
import os
import pickle

from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

parser = argparse.ArgumentParser(description='Website Q&A')
parser.add_argument('question', type=str, help='Your question for the inputed site')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

chain = VectorDBQAWithSourcesChain.from_llm(
        llm = OpenAI(model_name="gpt-4-32k",temperature=0.35,verbose=True), vectorstore=store, verbose=True)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
