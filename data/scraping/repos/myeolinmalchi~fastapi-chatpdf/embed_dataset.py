import os
import re
from typing import Any, Dict, Tuple

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from googletrans.client import Translated
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from kss import split_sentences
from utils import create_chunks
from langchain.embeddings import SentenceTransformerEmbeddings
from googletrans import Translator

class EmbeddingFailureException(Exception):
    def __init__(self):
        super().__init__('임베딩에 실패했습니다.')

def load_pdf_files(path):
    files = os.listdir(path)
    pdf_files = [file for file in files if file.endswith('.pdf')]
    return pdf_files

# 주어진 문자열을 패턴에 맞게 정제하여 반환
def refine_text(pattern: str, text: str) -> str:
    pattern_data = re.findall(pattern, text)
    regex_pattern = r"|".join([re.escape(item) for item in pattern_data])
    refined_text = re.sub(regex_pattern, "", text)

    return refined_text

# 주어진 문자열에서 불필요한 텍스트를 삭제하여 반환
def remove_dump(text: str) -> str:
    return refine_text(
        pattern=r'공개특허 \d+-\d+-\d+\n-\d+-|등록특허 \d+-\d+-\d+\n-\d+-|공개특허 \d+-\d+\n-\d+-|등록특허 \d+-\d+\n-\d+-', 
        text=text
    )

# 주어진 문자열에서 줄번호를 삭제하여 반환
def remove_line_n(text: str) -> str:
    return refine_text(
        pattern=r' \[\d+\]', 
        text=text
    )

# 주어진 문자열에서 메타데이터(IPC, 발명 명칭) 추출
def extract_metadata(text: str) -> Dict[str, Any]:
    IPC_pattern = r'[A-Z]\d+[A-Z]? \d+/\d+'
    IPC = re.findall(IPC_pattern, text)

    metadata = { "IPC": IPC, "patent_name": "" }
    patent_search_result = re.search(r"\(54\)([^()]+?)\n\(", text)
    if patent_search_result is not None:
        patent_name = patent_search_result.group(1).replace("발명의 명칭", "").replace("\n", " ")
        metadata["patent_name"] = patent_name

    return metadata

# 단일 pdf파일에서 텍스트와 메타데이터를 추출
def extract_pdf(file_path: str) -> Tuple[str, Dict[str, Any]]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    page_texts = [remove_dump(page.page_content) for page in pages]
    combined_text = " ".join(page_texts)
    result = remove_line_n(combined_text)
    metadata = extract_metadata(page_texts[0])

    return (result, metadata)

def embed_pdf(path, idx, file_name, embeddings, collection_name, client_settings, translator):

    file_path = os.path.join(path, file_name)
    text, metadata = extract_pdf(file_path)

    temp = split_sentences(text=text, strip=True)
    chunks = [chunk.replace("\n", " ") for chunk in create_chunks(temp, 1500)]

    chunks_eng = []

    for chunk in chunks:
        result = translator.translate(text=chunk, src="ko", dest="en")
        if isinstance(result, Translated):
            chunks_eng.append(result.text)

    Chroma.from_texts(
        texts=chunks_eng, 
        embedding=embeddings, 
        collection_name=collection_name, 
        metadatas=[
            {
                "doc_title": file_name, 
                "doc_id": idx, 
                "chunk_idx": i, 
                "origin_content": chunks[i]
                #"IPC": metadata["IPC"]
            } for i in range(len(chunks_eng))], 
        documents=[file_name for _ in range(len(chunks_eng))], 
        client_settings=client_settings
    )

def embed_dataset(
        path,
        chroma_client,
        collection_name,
        embeddings,
        client_settings, 
        translator 
    ):
    try:
        pdf_files = load_pdf_files(path=path)

        if collection_name in [collection.name for collection in chroma_client.list_collections()]:
            print(f"Collection '{collection_name}'이 이미 존재하여 삭제합니다.")
            chroma_client.delete_collection(name=collection_name)

        print(f"Collection '{collection_name}'을 생성합니다.")
        chroma_client.create_collection(name=collection_name)

        for idx, file_name in enumerate(pdf_files):
            print(f"Embedding [{idx+1} / {len(pdf_files)}] ... ", end="")
            embed_pdf(
                idx=idx, 
                file_name=file_name, 
                embeddings=embeddings, 
                collection_name=collection_name, 
                path=path, 
                client_settings=client_settings, 
                translator=translator
            )
            print("Done.")

        print("Complete.")

    except Exception as e:
        print(f"Exception: {e}")
        print(f"오류가 발생하여 Collection을 초기화합니다.")
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Exception: {e}")
            print("Collection 초기화에 실패하였습니다.")
            raise EmbeddingFailureException

def main():
    load_dotenv()

    CHROMA_DB_HOST = os.environ['CHROMA_DB_HOST']
    CHROMA_DB_PORT = os.environ['CHROMA_DB_PORT']

    translator = Translator()

    client_settings = Settings(
        chroma_api_impl="rest", 
        chroma_server_host=CHROMA_DB_HOST, 
        chroma_server_http_port=CHROMA_DB_PORT
    )

    print(f"Chroma DB에 연결합니다. [{CHROMA_DB_HOST}:{CHROMA_DB_PORT}]\n")

    chroma_client = chromadb.Client(client_settings)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    embed_dataset(
        path='./datasets', 
        chroma_client=chroma_client, 
        collection_name='test_collection', 
        embeddings=embeddings, 
        client_settings=client_settings, 
        translator=translator
    )

if __name__ == '__main__':
    main()
