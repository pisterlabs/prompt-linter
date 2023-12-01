import streamlit as st
from PyPDF2 import PdfReader
import config
import os

# 通过openai的embedding接口将文档转化为向量，并存入pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

# 获取配置信息
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_ENV = config.PINECONE_ENV
OPENAI_API_KEY = config.OPENAI_API_KEY
PINECONE_INDEX = config.PINECONE_INDEX

# 初始化embeddings接口
embeddings = OpenAIEmbeddings()

# 无需重复初始化pinecone接口
index_name = PINECONE_INDEX

st.sidebar.success("Upload Files")

# 设置页面标题
st.title("File upload")

# 设置文件名
filename = "upload_records.txt"

# 检查文件是否存在，如果不存在则创建
if not os.path.exists(filename):
    with open(filename, "w") as f:
        pass

uploaded_file = st.file_uploader("请选择一个PDF文件：", type="pdf")

if st.button("上传"):
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        st.write("PDF文件内容：")
        st.write(chunks)

        # 提示处理中，等待处理
        with st.spinner('Wait for vectorization...'):
            # pinecone.Index(index_name)
            # 索引导入一次即可
            docsearch = Pinecone.from_texts(chunks, embeddings, index_name=index_name)
        st.success('Done!')

        # 将上传记录写入文件
        with open(filename, "a") as f:
            f.write(f"{uploaded_file.name}\n")
            st.success(f"File {uploaded_file.name} uploaded successfully!")
