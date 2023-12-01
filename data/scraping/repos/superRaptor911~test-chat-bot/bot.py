from langchain.llms import OpenAI
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from utility import unpickle_object

# set up API key


def setup_bot(model_name="gpt-3.5-turbo"):
    print("Loading data")
    documents = unpickle_object("./output/documents.bin")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embeddings = OpenAIEmbeddings()  # pyright: ignore
    vectorstore = Chroma.from_documents(documents, embeddings)
    print("Loading complete")

    bot = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.7, model=model_name),  # pyright: ignore
        vectorstore.as_retriever(),
        memory=memory,
    )
    print("Bot ready")
    return bot


chat_history = []
bot = setup_bot("gpt-4")


def bot_response(user_message):
    response = bot({"question": user_message, "chat_history": chat_history})
    chat_history.append((user_message, response["answer"]))
    return response["answer"]


def get_history():
    return chat_history
