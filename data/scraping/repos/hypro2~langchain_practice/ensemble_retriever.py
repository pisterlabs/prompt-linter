import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

"""

EnsembleRetriever 검색기 리스트를 입력으로 가져 와서 결과를 앙상블 get_relevant_documents() 방법을 기반으로
결과를 다시 순위 상호 순위 융합 알고리즘입니다.
다른 알고리즘의 장점을 활용하여 EnsembleRetriever 단일 알고리즘보다 더 나은 성능을 달성 할 수 있습니다.

가장 일반적인 패턴은 스파 스 리트리버 (예 : BM25)와 조밀 한 리트리버 (예 : 유사성 포함)를 결합하는 것입니다. 
"하이브리드 검색"이라고도합니다". 스파 스 리트리버는 키워드를 기반으로 관련 문서를 찾는 데 능숙하고, 
조밀 한 리트리버는 의미론적 유사성을 기반으로 관련 문서를 찾는 데 능숙합니다.

"""

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS


documents = TextLoader('../dataset/state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)[:10]


embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
faiss_vectorstore  = FAISS.from_documents(documents=texts, embedding=embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.5, 0.5])

question = "What are the approaches to Task Decomposition?"
docs = ensemble_retriever.get_relevant_documents(question)
print(docs)