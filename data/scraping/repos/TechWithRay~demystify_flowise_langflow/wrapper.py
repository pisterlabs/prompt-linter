import os
import argparse
import logging
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Pinecone
from typing import List, Tuple, Union
from consts import PINECONE_SETTINGS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import pinecone
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--device_type",
    type=str,
    default="cpu",
    choices=["cuda", "cpu", "hip"],
    help="The compute power that you have",
)


def load_model():
    """
    Select a model from huggingface.
    """
    language_model = OpenAI(
        model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY
    )

    return language_model



def main():

    args = parser.parse_args()
    logging.info(f"Running on: {args.device_type}")

    logging.info("Creating embedding for the documents")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": args.device_type}
    )

    index_name = "flowise"

    logging.info("Consume Pinecone VectorDB ")
    pinecone.init(**PINECONE_SETTINGS)
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    # query = "turing test"
    # docs = vectorstore.similarity_search(query)
    # print(docs)

    llm = load_model()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    with open("user_input.log", "w") as file:
        while True:
            query = input("\nEnter a question: ")

            # Write the query to the log file
            file.write(query + "\n")

            if query == "quit":
                break

            answer = qa.run(query=query)

            # Print the answer
            print(f"\n\n > Question:")
            print(query)
            print(f"\n\n > Answer:")
            print(answer)


if __name__ == "__main__":
    try:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )
        main()
    except:
        raise RuntimeError("Something went wrong")
