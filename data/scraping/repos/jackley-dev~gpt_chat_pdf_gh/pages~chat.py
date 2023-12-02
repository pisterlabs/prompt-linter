import streamlit as st
import config
import os

# 通过openai的embedding接口将文档转化为向量，并存入pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# 获取配置信息
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_ENV = config.PINECONE_ENV
OPENAI_API_KEY = config.OPENAI_API_KEY
PINECONE_INDEX = config.PINECONE_INDEX

# 初始化embeddings接口
embeddings = OpenAIEmbeddings()

# 无需重复初始化pinecone接口
index_name = PINECONE_INDEX

# 初始化openai聊天接口
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

st.sidebar.success("Chat with your repository")

# 设置文件名
filename = "upload_records.txt"

# 检查文件是否存在，如果不存在则创建
if not os.path.exists(filename):
    with open(filename, "w") as f:
        pass

# 显示上传记录
st.write('uploaded_records:')
with open(filename, "r") as f:
    for line in f:
        st.write(line)

# 提交区域
txt = st.text_area(':sunglasses: 向知识库提问吧：', '''...''')

if st.button('Submit'):
    # 在向量数据库中，查询相似的文档
    # pinecone.Index(index_name)
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    query = txt
    # 问答记录区域
    st.markdown('---')
    with st.spinner('Waiting for AI searching similar docs...'):
        docs = docsearch.similarity_search(query, 3)
    st.success('Searching similar docs done!')
    st.write('搜索到的相似文档如下：')
    st.write(docs)
    # 提示AI处理中
    with st.spinner('Waiting for AI answering...'):
        ans = chain.run(input_documents=docs, question=query)
    st.success('AI answering done!')
    st.write('AI回复如下：')
    st.write(ans)