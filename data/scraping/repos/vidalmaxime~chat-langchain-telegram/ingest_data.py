import argparse
import pickle

from langchain.text_splitter import CharacterTextSplitter
from telegram_chat_loader import TelegramChatLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


# Load Data
def embed_chat(chat_file_path):
    loader = TelegramChatLoader(chat_file_path)
    raw_documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=512, chunk_overlap=20)
    documents = text_splitter.split_documents(raw_documents)

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='The Telegram chat exported *.json file')
    args = parser.parse_args()
    file_name = args.file_name
    embed_chat(file_name)
