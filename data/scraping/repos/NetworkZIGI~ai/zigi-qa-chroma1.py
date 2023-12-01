import os

from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader

import envinfo

## OpenAI를 사용하기 위한 변수 초기화 ##
AZURE_OPENAI_KEY = envinfo.openai_api_key
AZURE_OPENAI_ENDPOINT = envinfo.openai_api_base
AZURE_OPENAI_API_VERSION = envinfo.openai_api_version
AZURE_OPENAI_API_TYPE = envinfo.openai_api_type

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_KEY
os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION

# PDF 파일 경로 목록
pdf_files = [
    "./data/visa-kor.pdf",
]

# 로더, 텍스트 분할기 및 임베딩 초기화
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = OpenAIEmbeddings()
texts = []

# 각 PDF 파일에 대해 작업 수행
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    texts.extend(text_splitter.split_documents(documents))

docsearch = Chroma.from_documents(texts, embeddings)

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Korean:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

#deployment_model = 'gpt-35-turbo'
deployment_model = 'gpt-4'
llm = AzureChatOpenAI(deployment_name=deployment_model,
                      temperature=0.1, max_tokens=500)

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type='stuff', # end: stuff , kor : map_reduce
                                 retriever=docsearch.as_retriever(),
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs)

query = "비자카드 무료 발렛파킹 서비스 횟수는 얼마나 되는가?"
result = qa({"query": query})
answer = result['result']
print(f"질문 : {query}")
print(f"답변 : {answer}")
