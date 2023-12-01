from langchain.document_loaders import CSVLoader
# from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import config
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = config.openAI


def csv_to_bot(filename):

    text_loader = CSVLoader(file_path=filename)
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loader([text_loader])

    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    def chat():
        while True:
            message = input("Enter a message: ")
            if message == "quit":
                break
            else:
                res = chain({"question": message})
                print(res['result'])

    chat()


if __name__ == "__main__":
    csv_to_bot("pokemon.csv")
