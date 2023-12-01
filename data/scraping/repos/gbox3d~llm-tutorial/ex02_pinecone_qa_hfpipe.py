#%%
import os
import time
from operator import itemgetter

from transformers import AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import VectorDBQA
from langchain import OpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

import pinecone
from dotenv import load_dotenv

# .env 파일 로드 및 Pinecone 초기화
load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), 
              environment=os.getenv("PINECONE_ENVIRONMENT"))

#%% Hugging Face 모델 및 파이프라인 로드
model_name = os.getenv("HUGGINGFACE_MODEL")
print(f'Start loading {model_name}')

start_tick = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_name)

hf_pipeline = pipeline(
    "text-generation", 
    model=model_name, 
    tokenizer=tokenizer,
    max_length=1024,
    device_map="auto"  # GPU 사용 설정
)

print(f'Load time: {time.time() - start_tick}')

# HuggingFacePipeline 인스턴스 생성
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Pinecone 벡터 스토어 설정
embeddings = OpenAIEmbeddings()
vectordb = Pinecone.from_existing_index(
    embedding=embeddings,
    index_name=os.getenv("PINECONE_INDEX_NAME")
)
# retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
#%% QA 모델 설정
# qa_chain = VectorDBQA.from_chain_type(
#         llm=llm, chain_type="stuff", vectorstore=vectordb, return_source_documents=True
#     )

qa_chain = RetrievalQA.from_chain_type(llm=llm,
    chain_type="stuff",
    retriever=retriever,
    # chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True)

#%%
start_tick = time.time()
query = "What is a vector DB? Give me a 15 word answer for a begginner"
result = qa_chain({"query": query})

print(f'Query time: {time.time() - start_tick}')

# %%
print(f'question: {result["query"]}')
print(f'answer: {result["result"]}')
print(len(result['source_documents']))
# %%
