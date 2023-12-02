import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

"""

거리 기반 벡터 데이터베이스 검색은 고차원 공간에 쿼리를 포함 (대표)하고 "거리"를 기반으로 유사한 임베디드 문서를 찾습니다". 
그러나 검색은 쿼리 문구의 미묘한 변화로 또는 임베딩이 데이터의 의미를 잘 포착하지 못하는 경우 다른 결과를 생성 할 수 있습니다.

그만큼 MultiQueryRetriever LLM을 사용하여 주어진 사용자 입력 쿼리에 대해 다른 관점에서 여러 쿼리를 생성하여 프롬프트 튜닝 프로세스를 자동화합니다. 
각 쿼리에 대해 관련 문서 세트를 검색하고 모든 쿼리에서 고유 한 조합을 사용하여 잠재적으로 관련된 문서 세트를 더 많이 얻습니다.
같은 질문에 대해 여러 관점을 생성함으로써 MultiQueryRetriever 거리 기반 검색의 일부 한계를 극복하고 더 풍부한 결과를 얻을 수 있습니다.

"""


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

documents = TextLoader('../dataset/state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)[:10]


embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = FAISS.from_documents(documents=texts, embedding=embedding)


question = "What are the approaches to Task Decomposition?"
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(),
                                                  llm=llm)



unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print(unique_docs)