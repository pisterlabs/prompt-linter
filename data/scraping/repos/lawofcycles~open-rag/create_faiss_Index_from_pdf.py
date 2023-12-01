import argparse
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS

parser = argparse.ArgumentParser(description='引数のpdfファイルを読み込んでFAISSのインデックスを作成する')
parser.add_argument('arg1', type=str, help='ファイルパス')
parser.add_argument('arg2', type=str, help='生成するindexの名前')
args = parser.parse_args()

loader = DirectoryLoader(args.arg1)
documents = loader.load()
print(f"number of docs: {len(documents)}")
print("--------------------------------------------------")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder()

splitted_texts = text_splitter.split_documents(documents)
print(f"チャンクの総数：{len(splitted_texts)}")
print(f"1番目のチャンク：\n{splitted_texts[0]}")

# embed model
embed_model_id = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

db = FAISS.from_documents(splitted_texts, embeddings)
db.save_local("faiss_index/" + args.arg2)
