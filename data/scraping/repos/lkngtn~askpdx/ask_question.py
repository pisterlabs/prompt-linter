import argparse
import pickle 

from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

parser = argparse.ArgumentParser(description='Ask PDX: Charter, Code, and Policy Q&A')
parser.add_argument('question', type=str, help='Your question Ask PDX')
args = parser.parse_args()

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

#llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k')
llm = ChatOpenAI(temperature=0, model='gpt-4')

chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=store.as_retriever(), chain_type='stuff')

result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")