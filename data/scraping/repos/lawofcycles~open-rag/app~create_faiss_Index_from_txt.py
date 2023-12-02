from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import argparse

parser = argparse.ArgumentParser(description='引数のtxtファイルを読み込んでFAISSのインデックスを作成する')

# 引数の追加
parser.add_argument('arg1', type=str, help='txtファイルのパス')
parser.add_argument('arg2', type=str, help='indexの名前')
args = parser.parse_args()

loader = UnstructuredFileLoader(args.arg1)
documents = loader.load()
print(f"number of docs: {len(documents)}")
print("--------------------------------------------------")

# 文章を分割する際の仕様を設定
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=20,
    separators=["\n\n\n","\n\n","\n"],
)

splitted_texts = text_splitter.split_documents(documents)
print(f"チャンクの総数：{len(splitted_texts)}")
print(f"1番目のチャンク：\n{splitted_texts[0]}")

# embed model
embed_model_id = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

db = FAISS.from_documents(splitted_texts, embeddings)
db.save_local("faiss_index/" + args.arg2)
