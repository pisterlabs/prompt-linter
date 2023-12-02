"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse
from dotenv import load_dotenv, dotenv_values

#loading the keys from python-dotenv
load_dotenv()
config = dotenv_values(".env.local") #or .env for shared code


def qa(question:str):
    # Load the LangChain.
    index = faiss.read_index("docs.index")

    with open("faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
    result = chain({"question": args.question})
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    return result["answer"]

"""
Calling the function as a script just to test
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
    parser.add_argument('question', type=str, help='The question to ask the notion DB')
    args = parser.parse_args()
    qa(args.question)