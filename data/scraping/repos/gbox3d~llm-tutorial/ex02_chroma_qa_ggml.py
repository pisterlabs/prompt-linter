#%%
import os
import time
from operator import itemgetter

from transformers import AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline,CTransformers

from langchain.vectorstores import Chroma

from langchain import VectorDBQA
from langchain.chains import RetrievalQA

from langchain.embeddings.openai import OpenAIEmbeddings


from langchain import OpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from ctransformers import AutoModelForCausalLM
import torch
import pinecone
from dotenv import load_dotenv

# .env 파일 로드 및 Pinecone 초기화
load_dotenv()
#%%
# Pinecone 벡터 스토어 설정
embeddings = OpenAIEmbeddings()

# 저장된 벡터 저장소 로드
vector_store = Chroma(
    persist_directory='./stores/chroma_store',
    embedding_function=embeddings
                      )
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
print('vector store loaded')
#%%
# Hugging Face 모델 및 파이프라인 로드
model_name = 'TheBloke/Llama-2-7B-Chat-GGML'
llm = CTransformers(
    model=model_name,
    model_type='llama',
    max_length=4096,
    temperature=0.1,
    device_map="auto" # GPU 상황에 맞게 자동으로 설정
)
#%%

llm.predict('안녕하세요')
#%% QA chain 생성 
qa_chain = VectorDBQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        vectorstore=vector_store, 
        return_source_documents=True,
        k=1
    )

#%%
start_tick = time.time()
query = "정읍사 공원 에 대해서 알려줘"
result = qa_chain({"query": query})
print(f'Query time: {time.time() - start_tick}')

# %%
print(f'question: {result["query"]}')
print(f'answer: {result["result"]}')
print(f'document count : {len(result["source_documents"])}')
# %%
for doc in result['source_documents']:
    print(doc)
# %%
