import os
from dotenv import load_dotenv
import argparse

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

from module.vectorstore import VectorStore
from module.manualchat import ManualChat

load_dotenv()
index_name = "langchain-demo"
model_name = "gpt-3.5-turbo"

openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
vectorstore = VectorStore(index_name)
manualchat = ManualChat(vectorstore.get_vectorstore(embeddings), model_name)


def insert_pdf(path: str):
  loader = PyPDFLoader(path)
  pages = loader.load_and_split()
  vectorstore.insert_vectors(pages, embeddings)


def ask(question: str):
  result = manualchat.ask(question)
  return result["answer"]


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PDFをベクトルストアに挿入するか、チャットボットに質問する")
  parser.add_argument("--pdf", action="store_true", help="PDFをベクトルストアに挿入する")
  parser.add_argument("--question", action="store_true", help="チャットボットに質問する")
  args = parser.parse_args()

  # PDFをベクトルストアに挿入する
  if args.pdf:
    for file in os.listdir("./pdf"):
      if file.endswith(".pdf"):
        print(f"insert {file}")
        insert_pdf("./pdf/" + file)
      break

  # チャットボットに質問する
  if args.question:
    while True:
      question = input("あなたの質問：")
      if question == "exit":
        break
      ask(question)
