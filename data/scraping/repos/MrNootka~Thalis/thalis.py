import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import argparse

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
embedding = OpenAIEmbeddings()

db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 2})

turbo_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=turbo_llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

def process_llm_response(llm_response):
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])

def main():
    args = parse_arguments()

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        llm_response = qa_chain(query)
        process_llm_response(llm_response)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DocumentsGPT: Interact with your documents using OpenAI's API instead of local language models."
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()