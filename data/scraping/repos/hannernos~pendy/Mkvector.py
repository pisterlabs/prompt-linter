from openaikey import apikey
import os
import chromadb
os.environ["OPENAI_API_KEY"] = apikey


from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import BaseSettings
import chromadb


file_name = "../product.txt"
file=open(file_name,"r",encoding="utf-8")
# file.write()
# file.close()

plain_txt = file.read()

doc_chunks = []

for line in plain_txt:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, # 최대 청크 길이
        separators=["\n\n", "\n", "\t",".", "!", "?", ",", " ", ""], #  텍스트를 청크로 분할하는 데 사용되는 문자 목록
        chunk_overlap=0, # 인접한 청크 간에 중복되는 문자 수
    )
    chunks = text_splitter.split_text(line)
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk, metadata={"page": i, "source": file_name}
        )
        doc_chunks.append(doc)
		#persist_directory 을 명시하지 않으면 현재 디렉토리에 저장합니다
persist_directory = "./vector_chromadb"
#os_enviroment에 api_key를 입력했다면 파라미터로 전달하지 않아도 됩니다
embeddings = OpenAIEmbeddings(api_key = apikey)
vectordb = Chroma.from_documents(documents=doc_chunks,
embedding=embeddings, persist_directory=persist_directory)

#chroma db를 스토리지에 저장합니다
vectordb.persist()