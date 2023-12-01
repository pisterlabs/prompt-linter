import os
import platform

import openai
import chromadb
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# keiojp.py
chat_history = []
def process_message(message):
    # メッセージを処理し、返信するテキストを生成します
    query = message
    response_text = pdf_qa({"question": query, "chat_history": chat_history})
    return response_text["answer"]


# PDFファイルをロードする
loader = PyPDFLoader("data/main/submit.pdf")

# PDFからページを抽出し、指定したchunk_sizeでテキストを分割する
pages = (loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,  # テキストのチャンクサイズ
    chunk_overlap=20,  # チャンク間のオーバーラップサイズ
    length_function=len,  # テキストの長さを計算する関数
)))

# OpenAI APIキーを設定する
openai.apy_key = os.environ["OPENAI_API_KEY"]

# GPT-3.5-turboモデルを使用したチャットモデルを作成する
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# OpenAI Embeddingsを作成する
embeddings = OpenAIEmbeddings()

# ページの内容をもとにChroma VectorStoreを作成する
vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory="./database")

# VectorStoreを永続化する
vectorstore.persist()

# GPT-3.5-turboとVectorStoreを使って会話型検索チェーンを作成する
pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=False)
"""
# 質問を設定する
query = "卒業条件は何単位ですか?"

# チャット履歴を初期化する
chat_history = []

# 質問とチャット履歴を使って回答を取得する
result = pdf_qa({"question": query, "chat_history": chat_history})

# 回答を表示する
print(result["answer"])
"""
