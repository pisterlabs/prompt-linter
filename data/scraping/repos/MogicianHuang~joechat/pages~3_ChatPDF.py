import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import PyPDF2
from io import BytesIO
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import time
from langchain.chat_models import ChatOpenAI
### init

st.set_page_config(page_title="ChatPDF", page_icon="📄")

st.write("# 上传PDF文件，向ChatGPT进行提问")

if 'ChangePdfModel' not in st.session_state:
    st.session_state.ChangePdfModel = False
def change_model():
    st.session_state.ChangePdfModel = True

col1 , col2 = st.columns(2)
model = col1.selectbox(
    'Model',
    ('gpt-3.5-turbo','gpt-4'),
    on_change=change_model
)

temperature = col2.slider(
    'temperature', 0.0, 1.0, 0.8, step=0.01
)

choose_way = st.sidebar.selectbox(
    '文件获取方式',
    ('Upload', 'URL')
)

data = None
docsearch = None
human_query = None
llm = None

if 'pdf_history' not in st.session_state:
        st.session_state.pdf_history = []

### load pdf
if choose_way == 'Upload':
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open('output.pdf', 'wb') as f:
            f.write(uploaded_file.getvalue())
        uploaded_file = PyPDFLoader("output.pdf")
        data = uploaded_file.load()
        if data is not None:
            st.sidebar.success("文件上传成功")
else:
    target_url = st.sidebar.text_input("URL")
    if target_url:
        uploaded_file = OnlinePDFLoader(target_url)
        data = uploaded_file.load()
        if data is not None:
            st.sidebar.success("文件下载成功")

### load data to pinecone
if data is not None:
    docs_num = len(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    texts_num = len(texts)
    pinecone.init(      
	    api_key=os.environ["pinecone_api_key"],      
	    environment=os.environ["pinecone_env"]     
    )      
    index_name = "joehuang"

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api_key"], openai_api_base=st.session_state["openai_api_base"])

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)




### Q&A with history
if docsearch is not None:
    human_query = st.chat_input(f"向{model}进行提问")
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.info(f"文件分析完成，文章共有{docs_num}页，共分为了{texts_num}段，开始提问吧！")
for i, (query, response) in enumerate(st.session_state.pdf_history):
    with st.chat_message(name="user", avatar="user"):
        st.markdown(query)
    with st.chat_message(name="assistant", avatar="assistant"):
        st.markdown(response)

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
if human_query is not None:    
    input_placeholder.markdown(human_query)

if llm is None or st.session_state.ChangePdfModel:
    llm = ChatOpenAI(temperature=temperature, openai_api_key=st.session_state["openai_api_key"], openai_api_base=st.session_state["openai_api_base"],model=model)
    st.session_state.ChangePdfModel = False

if human_query is not None and docsearch is not None:
    history = st.session_state.pdf_history
    docs = docsearch.similarity_search(human_query)
    # st.write(docs[0].page_content)
    chain = load_qa_chain(llm, chain_type="stuff")
    ans = chain.run(input_documents=docs, question=human_query)
    with st.chat_message(name="assistant", avatar="assistant"):
        message_placeholder = st.write(ans)
    st.session_state.pdf_history.append((human_query,ans))

### clear history
clear_history = st.button("🧹", key="clear_history")
if clear_history:
    st.session_state.pdf_history.clear()

