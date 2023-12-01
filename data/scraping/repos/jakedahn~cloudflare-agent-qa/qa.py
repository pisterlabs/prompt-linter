import argparse
import pickle

import faiss

from promptlayer.langchain.llms import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain


parser = argparse.ArgumentParser(description="Ask a question to the cloudflare docs.")
parser.add_argument(
    "question", type=str, help="The question to ask the cloudflare docs"
)
args = parser.parse_args()


index = faiss.read_index("docs.index")

with open("cloudflare_docs.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(
    llm=OpenAI(
        temperature=0,
        max_tokens=1500,
        model_name="text-davinci-003",
        max_retries=3,
        pl_tags=["cloudflare-qa-agent"],
    ),
    vectorstore=store,
)
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
