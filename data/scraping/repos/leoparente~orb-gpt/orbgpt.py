#!/usr/bin/env python3

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from constants import PERSIST_DIRECTORY

if __name__ == "__main__":
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever=vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    print("Chat with ORB docs!")
    try:
        while True:
            print("\nQuestion:")
            question = input()
            if(question == "quit"):
                print("\nExiting Chat... See ya!")
                break 
            result = chat({"question": question})
            print("OrbGPT:")
            print(result["answer"])
    except KeyboardInterrupt:
        print("\nExiting Chat... See ya!")