import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import re
import os
import time
import pinecone
from typing import List, Union,Callable,Dict, Optional, Any
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader,UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
import asyncio

st.set_page_config(
    page_title="ChatReport",
    page_icon="https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.cpanlp.com/',
        'Report a bug': "https://www.cpanlp.com/",
        'About': "智能报告"
    }
)
def web_file_pdf(input_text):
    loader = PyPDFLoader(input_text)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    Pinecone.from_documents(texts, embeddings_cho, index_name="kedu",namespace=pinename)
    st.success(f"已上传来自 PDF 文件的 {len(texts)} 个文档。”")
    st.cache_data.clear()
def web_file_pptx(input_text):
    loader = UnstructuredPowerPointLoader(input_text)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    Pinecone.from_documents(texts, embeddings_cho, index_name="kedu",namespace=pinename)
    st.success(f"已上传来自 PPTX 文件的 {len(texts)} 个文档。”")
    st.cache_data.clear()
def web_file_docx(input_text):
    loader = Docx2txtLoader(input_text)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    Pinecone.from_documents(texts, embeddings_cho, index_name="kedu",namespace=pinename)
    st.success(f"已上传来自 DOCX 文件的 {len(texts)} 个文档。”")
    st.cache_data.clear()    

#     @st.cache_resource
def upload_file_pdf():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        loader = PyPDFLoader(tmp_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        Pinecone.from_documents(texts, embeddings_cho, index_name="kedu",namespace=pinename)
        st.success(f"已上传来自 PDF 文件的 {len(texts)} 个文档。”")
        st.cache_data.clear()
def upload_file_pptx():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        loader = UnstructuredPowerPointLoader(tmp_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        Pinecone.from_documents(texts, embeddings_cho, index_name="kedu",namespace=pinename)
        st.success(f"已上传来自 PPTX 文件的 {len(texts)} 个文档。”")
        st.cache_data.clear()
def upload_file_docx():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file.flush()
        loader = Docx2txtLoader(tmp_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        Pinecone.from_documents(texts, embeddings_cho, index_name="kedu",namespace=pinename)
        st.success(f"已上传来自 DOCX 文件的 {len(texts)} 个文档。”")
        st.cache_data.clear()
def upload_file():
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext == ".pptx":
        upload_file_pptx()
    elif file_ext == ".pdf":
        upload_file_pdf()
    elif file_ext == ".docx":
        upload_file_docx()
    else:
        st.warning("不支持的文件类型，请上传 PPTX、DOCX 或 PDF 文件。")
def web_file(input_text):
    if input_text.lower().endswith('.pptx'):
        web_file_pptx(input_text)
    elif input_text.lower().endswith('.pdf'):
        web_file_pdf(input_text)
    elif input_text.lower().endswith('.docx'):
        web_file_docx(input_text)
    else:
        st.warning("不支持的文件类型，请上传 PPTX 、DOCX 或 PDF 文件。")
def upload_query(input_file):
    ww=""
    pinecone.init(api_key="1ebbc1a4-f41e-43a7-b91e-24c03ebf0114",  # find at app.pinecone.io
          environment="us-west1-gcp-free", 
          namespace=pinename
          )
    index = pinecone.Index(index_name="kedu")
    a=embeddings_cho.embed_query(input_file)
    www=index.query(vector=a, top_k=top_k, namespace=pinename, include_metadata=True)
    for i in range(top_k):
        ww+=www["matches"][i]["metadata"]["text"]
    return ww

logo_url = "https://raw.githubusercontent.com/dengxinkai/cpanlp_streamlit/main/app/%E6%9C%AA%E5%91%BD%E5%90%8D.png"
with st.sidebar:
    with st.expander("👇 :blue[**第一步：输入 OpenAI API 密钥**]"):
        if 'input_api' in st.session_state:
            st.text_input("api-key",st.session_state["input_api"], key="input_api")
        else:
            st.info('请先输入正确的OpenAI API 密钥')
            st.text_input('api-key','', key="input_api",type="password")
        st.write("ChatOpenAI属性配置")
        temperature = st.slider("`temperature`", 0.01, 0.99, 0.3,help="用于控制生成文本随机性和多样性的参数。较高的温度值通常适用于生成较为自由流畅的文本，而较低的温度值则适用于生成更加确定性的文本。")
        frequency_penalty = st.slider("`frequency_penalty`", 0.01, 0.99, 0.3,help="用于控制生成文本中单词重复频率的技术。数值越大，模型对单词重复使用的惩罚就越严格，生成文本中出现相同单词的概率就越低；数值越小，生成文本中出现相同单词的概率就越高。")
        presence_penalty = st.slider("`presence_penalty`", 0.01, 0.99, 0.3,help="用于控制语言生成模型生成文本时对输入提示的重视程度的参数。presence_penalty的值较低，模型生成的文本可能与输入提示非常接近，但缺乏创意或原创性。presence_penalty设置为较高的值，模型可能生成更多样化、更具原创性但与输入提示较远的文本。")
        top_p = st.slider("`top_p`", 0.01, 0.99, 0.9,help="用于控制生成文本的多样性，较小的top_p值会让模型选择的词更加确定，生成的文本会更加一致，而较大的top_p值则会让模型选择的词更加多样，生成的文本则更加多样化。")
        model = st.radio("`模型选择`",
                                ("gpt-3.5-turbo",
                                "gpt-4"),
                                index=0,key="main_model")
    llm=ChatOpenAI(
        model_name=model,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        top_p=top_p,
        openai_api_key=st.session_state.input_api
    )
    with st.expander("👇 :blue[**第二步：创建自己的数据库或连接到已有数据库**]"):
        pinename = st.text_input('**数据库名称**','report',key="pinename",help="系统每天定期清理数据库")
        pinecone.init(api_key="1ebbc1a4-f41e-43a7-b91e-24c03ebf0114",  # find at app.pinecone.io
                          environment="us-west1-gcp-free", 
                          namespace=pinename
                          )
        index = pinecone.Index(index_name="kedu")
        if st.button('删除数据库',key="deletepine"):
            index = pinecone.Index(index_name="kedu")
            index.delete(deleteAll='true', namespace=pinename)
        st.warning("⬆️ 别忘了删除不再使用的数据库")
    with st.expander("👇 :blue[**第三步：选择数据库文件上传方式**]"):
        fileoption = st.radio('**数据库创建方式**',('本地上传', 'URL'),key="fileoption")
        chunk_size = st.number_input('chunk_size',value=800,min_value=200,max_value=2500,step=100,key="chunk_size",help='每个文本数据块的大小。例如，如果将chunk_size设置为1000，则将输入文本数据分成1000个字符的块。')
        chunk_overlap = st.number_input('chunk_overlap',value=0,min_value=0,max_value=500,step=50,key="chunk_overlap",help='每个文本数据块之间重叠的字符数。例如，如果将chunk_overlap设置为200，则相邻的两个块将有200个字符的重叠。这可以确保在块之间没有丢失的数据，同时还可以避免重复处理相邻块之间的数据。')
        top_k = st.number_input('top_k',value=3,min_value=0,max_value=10,step=1,key="top_k",help="用于控制查询的结果数量，指定从数据库中返回的与查询向量最相似的前 k 个向量")
        embeddings_cho = OpenAIEmbeddings(openai_api_key=st.session_state.input_api)
    with st.expander("👇 :blue[**第四步：数据上传**]"):
        if fileoption=="本地上传":
            file = st.file_uploader("上传文件（支持格式包括：PPTX、DOCX和PDF）", type=("pptx",'pdf','docx'),key="upload")
            if file is not None:
                with st.spinner('Wait for it...'):
                    upload_file()
        else:
            input_text = st.text_input('文件网址（支持格式包括：PPTX、DOCX 和 PDF）', 'http://static.cninfo.com.cn/finalpage/2023-04-29/1216712300.PDF',key="webupload")
            if st.button('载入数据库',key="pdfw"):
                with st.spinner('Wait for it...'):
                    pinecone.init(api_key="1ebbc1a4-f41e-43a7-b91e-24c03ebf0114",  # find at app.pinecone.io
                          environment="us-west1-gcp-free", 
                          namespace=pinename
                          )
                    index = pinecone.Index(index_name="kedu")
                    web_file(input_text)
                    st.cache_data.clear()
@st.cache_data(persist="disk")
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')
if st.button('刷新页面',key="rerun"):
    st.experimental_rerun()
    st.cache_data.clear()

if st.session_state.input_api:
    do_question=[]
    do_answer=[]
    with get_openai_callback() as cb:  
#主要功能                
        input_files = st.text_input('**批量查询**','#公司名称#公司产品',key="file_webss",help="不同问题用#隔开，比如：公司收入#公司名称#公司前景")
        if st.button('数据库批量查询',key="file_uploads1"):
            input_list = re.split(r'#', input_files)[1:]
            async def upload_all_files_async(input_list):
                do_question, do_answer = [], []
                tasks = []
                for input_file in input_list:
                    task = asyncio.create_task(upload_query_async(input_file))
                    tasks.append(task)
                results = await asyncio.gather(*tasks)
                with st.expander("数据库查询结果", expanded=True):
                    for key, inter_result in zip(input_list, results):
                        st.write(key)
                        st.success(inter_result)
                        do_question.append(key)
                        do_answer.append(inter_result)
                return do_question,do_answer
            async def upload_query_async(input_file):
                ww=""
                pinecone.init(api_key="1ebbc1a4-f41e-43a7-b91e-24c03ebf0114",  # find at app.pinecone.io
                              environment="us-west1-gcp-free", 
                              namespace=pinename
                              )
                index = pinecone.Index(index_name="kedu")
                a=embeddings_cho.embed_query(input_file)
                www=index.query(vector=a, top_k=top_k, namespace=pinename, include_metadata=True)
                for i in range(top_k):
                    ww+=www["matches"][i]["metadata"]["text"]
                return ww
            do_question, do_answer=asyncio.run(upload_all_files_async(input_list))
        if st.button('AI批量查询',key="aifile_uploadss",type="primary"):
            with st.spinner('Wait for it...'):
                start_time = time.time()
                input_list = re.split(r'#', input_files)[1:]
                async def upload_all_files_async(input_list):
                    do_question, do_answer = [], []
                    tasks = []
                    for input_file in input_list:
                        task = asyncio.create_task(upload_query_async(input_file))
                        tasks.append(task)
                    results = await asyncio.gather(*tasks)
                    with st.expander("AI查询结果", expanded=True):
                        for key, inter_result in zip(input_list, results):
                            st.write(key)
                            st.success(inter_result)
                            do_question.append(key)
                            do_answer.append(inter_result)
                    return do_question,do_answer
                async def upload_query_async(input_file):
                    ww=""
                    ww1=""
                    pinecone.init(api_key="1ebbc1a4-f41e-43a7-b91e-24c03ebf0114",  # find at app.pinecone.io
                          environment="us-west1-gcp-free", 
                          namespace='ceshi'
                          )
                    index = pinecone.Index(index_name="kedu")
                    start_time = time.time()
                    a=embeddings_cho.embed_query(input_file)
                    www=index.query(vector=a, top_k=top_k, namespace=pinename, include_metadata=True)
                    for i in range(top_k):
                        ww+=www["matches"][i]["metadata"]["text"]
                    template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
                    Return any relevant text verbatim.
                    Respond in Chinese.
                    QUESTION: {question}
                    =========
                    {summaries}
                    =========
                    FINAL ANSWER IN CHINESE:"""
                    prompt = PromptTemplate(
                        input_variables=["summaries", "question"],
                        template=template,
                    )
                    chain = LLMChain(prompt=prompt, llm=llm)
                    ww1=chain.predict(summaries=ww, question=input_file)
                    return ww1
                do_question, do_answer=asyncio.run(upload_all_files_async(input_list))
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"项目完成所需时间: {elapsed_time:.2f} 秒")  
                with st.expander("费用"):
                        st.success(f"Total Tokens: {cb.total_tokens}")
                        st.success(f"Prompt Tokens: {cb.prompt_tokens}")
                        st.success(f"Completion Tokens: {cb.completion_tokens}")
                        st.success(f"Total Cost (USD): ${cb.total_cost}")
        st.warning("⬆️ 使用数据库查询只需要通过 API 接口获取嵌入向量,使用 AI 查询需要使用 API 接口回答问题，并且会产生一定费用")
        df_inter = pd.DataFrame({
            '问题':do_question,
            '回答':do_answer,
             })
        if do_answer:
            with st.expander("回答记录"):
                st.dataframe(df_inter, use_container_width=True)
            csv_inter = convert_df(df_inter)
            st.download_button(
               "下载回答记录",
               csv_inter,
               "file.csv",
               "text/csv",
               key='download-csv_inter'
            )
            
else:
    st.header("请先输入正确的Openai api-key")
    
