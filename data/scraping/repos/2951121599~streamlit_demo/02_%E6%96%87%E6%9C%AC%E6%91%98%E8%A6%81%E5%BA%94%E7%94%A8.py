import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


# 根据给定的文本输出摘要
def generate_response(txt):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)


# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')  # 设置页面标题
st.title('🦜🔗 Text Summarization App')  # 显示页面标题

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):  # clear_on_submit=True在成功执行 AI 生成的响应后，我们使用参数清除 API 密钥文本框。
    # 用户通过st.text_input()方法指定的变量中存储的 OpenAI API 密钥
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            del openai_api_key  # 删除 API 密钥

if len(result):
    st.info(response)
