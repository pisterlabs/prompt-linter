from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import gradio as gr
import random
import time
import os
import json

load_dotenv()
embeddings = HuggingFaceEmbeddings()

# 向量数据库存储的路径
vectorstores_path = "../vectorstores"

# 已学习的文件列表
ingest_files_path = "../ingest_files.json"

vectorstore = None
if os.path.exists(vectorstores_path):
    vectorstore = FAISS.load_local(vectorstores_path, embeddings)

ingest_files = {}
if os.path.exists(ingest_files_path):
    with open(ingest_files_path, "r", encoding="utf-8") as f:
        ingest_files = json.load(f)

template = """
例子1: 
=========
已知内容:
问题: golang有哪些优势?

回答: 我不知道

例子2: 
=========   
已知内容:       
Content: 简单的并发
Source: 28-pl
Content: 部署方便
Source: 30-pl

问题: golang有哪些优势?

回答: 部署方便
SOURCES: 28-pl

例子3: 
=========
已知内容:
Content: 部署方便
Source: 0-pl

问题: golang有哪些优势?

回答: 部署方便
SOURCES: 28-pl

例子4:
=========
已知内容:
Content: 简单的并发
Source: 0-pl
Content: 稳定性好
Source: 24-pl
Content: 强大的标准库
Source: 5-pl

问题: golang有哪些优势?

回答: 简单的并发, 稳定性好
SOURCES: 0-pl,24-pl

=========
要求: 1. 参考上面的例子，回答如下问题; 在答案中总是返回 "SOURCES" 信息
要求: 2. 如果你不知道，请说 "抱歉，目前我还没涉及相关知识，无法回答该问题"
要求: 3. 如果你知道，尽可能多的回复用户的问题

已知内容:
{summaries}

问题: {question} 

使用中文回答:  
"""

PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
with gr.Blocks(css="static/main.css", title="询问智库科研助手") as demo:
    gr.Markdown("## 询问智库科研助手", elem_classes="title")
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=1, min_width=100):   
                    gr.HTML("""
                    <img src="file/static/logo.png"></img>
                    """, elem_classes='logo')
                with gr.Column(scale=8):
                    search_input = gr.Text(show_label=False, placeholder="问你想问的", elem_classes="search-input").style(container=False)
                with gr.Column(scale=1, min_width=80):
                    search_button = gr.Button("搜索", elem_classes="search-button")
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    pass
                with gr.Column(scale=9):
                    anwser_box = gr.Box(visible=False)
                    with anwser_box:
                        answer = gr.Markdown(show_label=False, interactive=False)

                    gr.Examples(label="您可以试着问我如下问题", examples=["询问智库是干嘛用的，对科学研究有什么帮助？", "请告诉我如何进行软件开发的研究工作？", "青藤云主机入侵检测产品有哪些常见功能？"], inputs=[search_input])
        with gr.Column(scale=1, min_width=300):
            with gr.Accordion("当前已学习资料"):
                dataset = gr.Dataset(label="",components=[gr.Textbox(visible=False)], samples=[[item] for item in ingest_files], elem_classes="docs").style(container=True)
            with gr.Accordion("帮我们完善智库，可以上传后提问"):
                upload_files = gr.Files(label='请上传资料文件', show_label=True, file_types=['.pdf'], visible=False, interactive=False)
                upload_button = gr.UploadButton("点击上传给智库", file_types=[".pdf"], file_count="multiple")

    def refresh_component():
        return dataset.update(samples = [[item] for item in ingest_files], visible=True)
    
    def search_question(content):
        chain_type_kwargs = {"prompt": PROMPT,"verbose": True}

        if vectorstore is None:
            yield answer.update("智库内容不存在，请先上传资料"), anwser_box.update(visible=True)
            return
        
        chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(max_tokens=-1, temperature=0), chain_type_kwargs = chain_type_kwargs, chain_type="stuff", retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": 0.2}))
        yield answer.update("请容我想一会儿..."), anwser_box.update(visible=True)
        result = chain({"question": content})
        print(result)
        output = f"""{result['answer']}"""
        if result['sources'] and len(result['sources']) > 0:
            output = f"""
            {result['answer']}

            **来源：{result['sources']}**
            """
        yield answer.update(output), anwser_box.update(visible=True)

    def ingest_upload_files(file_list):
        global vectorstore, ingest_files
        text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
        yield upload_files.update(None, visible=True, label=f"发现{len(file_list)}个文件，开始学习中，请稍等"), upload_button.update(visible=False), dataset.update(samples=[[item] for item in ingest_files],visible=True)
        file_paths = []
        for file in file_list:
            file_paths.append(file.name)
            docs = PyPDFLoader(file.name).load_and_split(text_splitter)
            documents = []
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file.name)
                documents.append(doc)
            if vectorstore is None:
                vectorstore = FAISS.from_documents(documents, embeddings)
            else:
                vectorstore.add_documents(documents)
            vectorstore.save_local(vectorstores_path)
            ingest_files[os.path.basename(file.name)] = True
            with open(ingest_files_path, "w", encoding="utf-8") as f:
                json.dump(ingest_files, f)

            yield upload_files.update(file_paths, visible=True, label=f"已完成如下资料学习，进度{len(file_paths)}/{len(file_list)}"), upload_button.update(visible=False), dataset.update(samples = [[item] for item in ingest_files], visible=True)
        yield upload_files.update(file_paths, visible=True, label='已完成您给的所有资料学习'), upload_button.update(visible=True), dataset.update(samples = [[item] for item in ingest_files], visible=True)
    
    demo.load(refresh_component, inputs=None, outputs=[dataset]) # 每次页面刷新时执行
    search_input.submit(search_question, inputs=search_input, outputs=[answer, anwser_box], show_progress=False)
    search_button.click(search_question, inputs=search_input, outputs=[answer, anwser_box], show_progress=False)
    # upload_button.DEFAULT_TEMP_DIR = "../upload-docs"     
    upload_button.upload(ingest_upload_files, inputs=upload_button, outputs=[upload_files, upload_button, dataset])

# 默认gradio可以访问启动路径下的所有文件，所以这里建立一个app目录用于启动脚本，防止一些敏感文件泄漏
# cd ./app && python ./main.py
demo.queue(concurrency_count=10)
demo.launch(share=True, favicon_path="static/favicon.ico", auth=("admin", "wikidb@123.com"), blocked_paths=["./main.py"])
