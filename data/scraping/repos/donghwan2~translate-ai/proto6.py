# [배포 시에만] chromadb의 sqlite3 버전 문제 해결 
# requirements.txt 에 pysqlite3-binary 추가
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# proto6 :
# 대화 history가 저장되고,
# 사용자가 미리 저장한 Chroma DB의 데이터를 이해한 상태에서,
# 사용자의 질문에 대한 답변을 "langchain.RetrievalQA"으로 답변"하는 챗봇

import streamlit as st

# api key
# from dotenv import load_dotenv
# load_dotenv()
import openai
openai.api_key = st.secrets["OPENAI_API_KEY"]

# llm : langchain.ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


### ★★★ 헤드 ★★★
st.markdown("# Translate-ai")


### st.session_state에 대화 내용 저장

# 모델 초기화 with st.session_state
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"         # LLM 모델 설정 : "gpt-3.5-turbo", "gpt-4"

# 대화 초기화 with st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 역할(role)과 대화내용(content) key 초기화
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


### Chroma db에 임베딩된 데이터 저장

# 주어진 문서를 로드하기
loader = DirectoryLoader('./statistics', loader_cls=TextLoader)   # glob="*.txt", 
documents = loader.load()

# Split texts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

# db(chroma db)를 저장할 디렉토리 지정
persist_directory = 'db'

# 임베딩
embedding = OpenAIEmbeddings()

# db에 임베딩된 데이터 저장
db = Chroma.from_documents(
    documents=texts, 
    embedding=embedding, 
    persist_directory=persist_directory) 

# db 초기화
db.persist()
db = None

db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding)


### 사용자 입력이 들어왔을 때, 사용자와 챗봇의 대화 생성

# ★★★ 사용자 인풋 창 ★★★
if question := st.chat_input("What is up?"):   # ★★★ 사용자 인풋 창 ★★★
    
    # 사용자 입력을 st.session_state에 저장
    st.session_state.messages.append({"role": "user", "content": question})   

    # ★★★ 사용자 아이콘 ★★★
    with st.chat_message("user"):     
        # ★★★ 사용자 입력을 마크다운으로 출력
        st.markdown(question)        
        print("question:", question, "\n")

    # 챗봇 대답
    # ★★★ 챗봇 아이콘 ★★★
    with st.chat_message("assistant"):

        # 챗봇 대답 생성 : 빈 placeholder 생성 후 한 줄씩 채워가기
        message_placeholder = st.empty()    # 빈 placeholder 생성
        full_response = ""
    
        # ＠＠＠ 챗봇 대답(full_response->str) 생성 모델링

        # 답변 생성 모델 : langchain.RetrievalQA
        # db : ChromaDB

        ## db에서 유사한 문서 찾기(결과를 k개 반환)
        retriever = db.as_retriever(search_kwargs={"k": 2})

        ## RetrievalQA 구성하기
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(), 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True)

        def process_llm_response(llm_response):
            print(llm_response['result'])
            print('\n\nSources:')
            for source in llm_response["source_documents"]:
                print(source.metadata['source'])
            print("\n")
            

        def qa_bot(query):
            llm_response = qa_chain(query)
            process_llm_response(llm_response)
            return llm_response['result']

        full_response = qa_bot(question)

        # ＠＠＠ 

        # ★★★ full_response(전체 답변 string)을 화면에 출력하기
        message_placeholder.markdown(full_response)

    # 생성된 챗봇 답변을 st.session_state에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    message_placeholder.markdown(full_response)
    print("session_state: \n", st.session_state, "\n")


# question 예시
# pando가 뭐야?
# 평균을 구하는 파이썬 코드를 알려줘
# 다음 글을 영어로 번역해줘. 
# 나는 아침에 차를 마셨더니 기분이 차차 나아졌어. 
# 점심에는 차를 타고 학교에 가서 여자친구에게 차였어. 
# 저녁에는 차병원에 가서 진료를 받았어





















