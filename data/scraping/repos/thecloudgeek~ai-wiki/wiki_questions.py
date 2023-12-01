from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pickle
import sys

OpenAI.openai_api_key = "key goes here"

chain = load_qa_with_sources_chain(OpenAI(temperature=0.0, model_name="gpt-3.5-turbo"),chain_type="stuff")

def print_answer(question):
    with open("search_index.pickle", "rb") as f:
        search_index = pickle.load(f)
    print(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=5),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )

question = sys.argv[1]

print_answer(question)