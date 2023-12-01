from langchain.chains import RetrievalQA
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import qdrant_client


def main():
    # Load stored vectorstore
    llama = LlamaCppEmbeddings(
        model_path="../models/ggml-model-q8_0.bin")

    client = qdrant_client.QdrantClient(
        path="./db",
        prefer_grpc=True
    )

    qdrant = Qdrant(
        client=client,
        collection_name="test",
        embeddings=llama
    )

    retriever = qdrant.as_retriever()
    # Prepare the LLM
    local_path = '../models/ggml-model-q8_0.bin'
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=local_path, callback_manager=callback_manager, verbose=True
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True)

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


if __name__ == "__main__":
    main()
