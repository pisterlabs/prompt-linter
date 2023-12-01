import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from elevenlabslib import *

# init elevenlabs
speak = True
user = ElevenLabsUser("elevenlabskey")
voice = user.get_voices_by_name("Geert")[0]


# get openai key from environment
OpenAI.openai_api_key = os.getenv('OPENAI_API_KEY')

# initialize wrapper around openai embeddings
# the embeddings are either loaded from local storage (faiss_index folder)
# or created from scratch
embeddings = OpenAIEmbeddings()

# try to load embeddings from local, otherwise re-create them
try:
    db = FAISS.load_local("faiss_index", embeddings)
    print("Loaded embeddings from local storage.")
except Exception as e:
    print(e)
    print("Could not load embeddings from local storage. Creating new embeddings.")

    # Load documents from a folder
    loader = TextLoader("./docs/1984.txt")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    # Create embeddings for chunks
    db = FAISS.from_documents(documents, embeddings)

    # Save embeddings to local storage
    db.save_local("faiss_index")

# Initialize the ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), db.as_retriever(search_kwargs={"k": 5}), return_source_documents=True)

def ask_question(query, chat_history=None):
    if chat_history is None:
        chat_history = []
    result = qa({"question": query, "chat_history": chat_history})

    return result["answer"]


print("Welcome to the Chat Interface! Type 'exit' to quit.")
chat_history = []
chat_history.append({"role": "system", "content": "You are a literary assistant for the book 1984! Only answer questions about the book"})

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    answer = ask_question(user_input, chat_history)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})
    print("Assistant:", answer)

    if speak:
        voice.generate_and_play_audio(answer, playInBackground=False)

    # pretty print chat history
    print("Chat History:")
    for item in chat_history:
        print(f"{item['role']}: {item['content']}")
    print("")
