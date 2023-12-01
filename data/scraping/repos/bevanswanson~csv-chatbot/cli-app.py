import os

from langchain.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_PATH = ""  # INSERT LOCAL MODEL PATH
FILE_PATH = ""  # INSERT LOCAL FILE PATH


def arrange_chatbot():
    loader = CSVLoader(
        file_path=FILE_PATH, encoding="utf-8", csv_args={"delimiter": ","}
    )
    data = loader.load()

    embeddings = GPT4AllEmbeddings()
    vectorstore = FAISS.from_documents(data, embeddings)
    callbacks = [StreamingStdOutCallbackHandler()]

    return ConversationalRetrievalChain.from_llm(
        llm=GPT4All(model=MODEL_PATH, n_threads=8, verbose=True, callbacks=callbacks),
        retriever=vectorstore.as_retriever(),
    )


def main():
    chain = arrange_chatbot()

    os.system("clear")

    while True:
        chat_history = []
        user_input = input("\nWhat ya sayin'?\n")
        if user_input == "exit":
            break
        result = chain.run({"question": user_input, "chat_history": chat_history})
        chat_history.append((user_input, result))


if __name__ == "__main__":
    main()
