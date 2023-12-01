from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
import logging
import gradio as gr
import json
from env import ini_env


#构建向量库index
def construct_index(folder_path,temperature,max_input_size,num_outputs,max_chunk_overlap,chunk_size_limit,folder_output_path):

    #设置模型
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    #读取目录下的文档
    documents = SimpleDirectoryReader(folder_path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.storage_context.persist(persist_dir=folder_output_path)

    #读取保存后的结果
    datastr = read_storage_data(folder_output_path)
    return "向量库建立成功：\n"+datastr;


#读取保存的向量库
def read_storage_data(folder_output_path):
    # 打开你的文件
    with open(folder_output_path+'/docstore.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 现在 'data' 是一个Python字典，它包含了你的JSON文件中的数据
    # 可以打印出来查看
    # 将Python字典进行格式化
    formatted_data = json.dumps(data, indent=4, ensure_ascii=False)

    # 输出格式化后的数据
    return formatted_data



def BuildDig():
    
    #设置一个对话窗

    folder_path = gr.inputs.Textbox(label="请输入文档目录",default="./docs")
    temperature_slider = gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.7, label="温度")
    max_input_size = gr.inputs.Slider(minimum=512, maximum=8192, default=4096, step=512, label="最大输入长度")
    num_outputs = gr.inputs.Slider(minimum=64, maximum=1024, default=512, step=64, label="输出长度")
    max_chunk_overlap = gr.inputs.Slider(minimum=10, maximum=50, default=20, step=5, label="最大分块重叠单词数")
    chunk_size_limit = gr.inputs.Slider(minimum=200, maximum=1000, default=600, step=100, label="分块大小限制")
    folder_output_path = gr.inputs.Textbox(label="请选择文档目录",default="./storage")
    demo = gr.Interface(
        construct_index,
        [folder_path,temperature_slider,max_input_size,num_outputs,max_chunk_overlap,chunk_size_limit,folder_output_path],
        ["text"],
        # 设置没有保存数据的按钮
        allow_flagging="never",
    )
    return demo

#加载环境变量
ini_env()
#启动服务
BuildDig().launch(share=True,server_port=17860,server_name="127.0.0.1")