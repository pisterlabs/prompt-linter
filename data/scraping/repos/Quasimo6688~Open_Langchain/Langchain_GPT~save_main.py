import numpy as np
import time
import os
import logging  # 用于日志功能
import configparser  # 用于读取配置文件
import nltk
import getpass
import json
from nltk.corpus import wordnet
# Langchain 相关导入
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
# 假设这些是可能需要的 Langchain Agents 和其他组件
from langchain.agents import OpenAIFunctionsAgent# 用于与语言模型交互

#谷歌搜索功能加载项
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory #内存记忆模块

#导入Gradio模块相关内容
import langchain_gradio_chat_interface
from langchain_gradio_chat_interface import GlobalState
# 创建全局状态实例
global_state = GlobalState()
global_state.finish_answer = ""
# 获取当前脚本的绝对路径的目录部分
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径来确定其他文件的绝对路径
api_key_file_path = os.path.join(script_dir, 'key.txt') #存储OPAI_API_KEY的文档
faiss_index_path = os.path.join(script_dir, 'faiss_index.index')#faiss索引文件
embeddings_path = os.path.join(script_dir, 'embeddings.npy')#Langchain知识库嵌入文件
metadata_path = os.path.join(script_dir, 'metadata.json')#知识库元数据

# 初始化日志和配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
global_log_output_str = ""

# 自动填写OpenAI API
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")

#初始化Open_AI
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=api_key, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

# 初始化变量
REQUEST_DELAY_SECONDS = 1
DEBUG = True  # 用于控制是否打印日志

streaming_active = False # 流式输出状态标示

def llm_to_UI():
    global_state.log_output_str = ""
    global_state.streaming_active = True
    prev_answer = None
    consecutive_no_changes = 0
    global_log_output_str = ""  # 重置日志字符串
    while global_state.streaming_active:
        system_msg = [HumanMessage(content=global_state.text_input)]
        gpt_response = chat(messages=system_msg)
        global_state.finish_answer = gpt_response.content

        log_message = f"模型回复: {gpt_response.content}"
        logging.info(log_message)
        global_state.log_output_str += log_message + "\n"

        # 检查global_finish_answer是否有变化
        if prev_answer == global_state.finish_answer:
            consecutive_no_changes += 1
        else:
            prev_answer = global_state.finish_answer
            consecutive_no_changes = 0

        # 如果连续3次没有变化并且答案不为空，我们停止生成器
        if consecutive_no_changes >= 3 and global_state.finish_answer:
            global_state.streaming_active = False

        yield global_state.finish_answer, global_state.log_output_str

        # 在每次检查之前等待1秒
        time.sleep(1) #在每次迭代之前等待2秒，然后检查global_finish_answer是否有变化。如果连续3次没有变化并且答案不为空，我们就停止生成器。这种方法应该可以缓解由于模型运算延迟或网络延迟引起的问题。

#执行Gradio模块的界面启动函数
langchain_gradio_chat_interface.start_UI(llm_to_UI, global_state)
