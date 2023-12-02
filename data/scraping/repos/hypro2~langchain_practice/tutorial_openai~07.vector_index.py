import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

"""
벡터 인덱스
쿼리와 가장 유사한 문장을 찾아서, 보다 정확한 답변을 얻을 수 있게 찾는 기능입니다. 
여기서는 Faiss라는 라이브러리를 이용해서 사용하는 예제를 보여줍니다.
메타데이터를 통해 보다 문서가 어떤 내용을 갖고 있는지 추가적인 정보를 전달 해줄 수 있습니다. 
similarity_search를 통해 코사인 유사도를 비교하고 k개 만큼의 문서를 사용합니다.
"""

# 데이터 준비
with open('../dataset/akazukin_all.txt', encoding='utf-8') as f:
    akazukin_all = f.read()

# 청크 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 청크의 최대 문자 수
    chunk_overlap=20,  # 최대 오버랩 문자 수
)
texts = text_splitter.split_text(akazukin_all)

# 확인
print(len(texts))
for text in texts:
    print(text[:10], ":", len(text))

# 메타데이터 준비
metadatas = [
    {"source": "1장"},
    {"source": "2장"},
    {"source": "3장"},
    {"source": "4장"},
    {"source": "5~6장"},
    {"source": "7장"}
]

# Faiss 벡터 인덱스 생성
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_texts(texts=texts,  # 청크 배열
                             embedding=embeddings,  # 임베딩
                             metadatas=metadatas  # 메타데이터
                             )

def search(query):
    docs = docsearch.similarity_search(query, k=3)
    print(docs[0].page_content)

if __name__=="__main__":
    query="미코의 소꿉친구 이름은?"
    search(query)