import openai
import time
import queue
import logging #加载日志功能
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
import threading
from state_manager import shared_output


# 配置日志记录器
logger = logging.getLogger(__name__)

class CustomStreamingCallback(StreamingStdOutCallbackHandler):#流式输出回调类
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.full_response = ""  # 用于存储整个响应的字符串

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)  # 将新令牌放入队列
        self.full_response += token  # 将新令牌附加到完整响应字符串
        logger.info(f"模型输出: {self.full_response}")  # 实时记录流式输出的当前状态

    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)  # 当输出结束时，将 None 放入队列
        logger.info(f"模型输出完成: {self.full_response}")  # 记录整个响应
        self.full_response = ""  # 重置完整响应字符串##


def initialize_model(api_key, model_name="gpt-3.5-turbo", temperature=0.5, streaming=True):
    streaming_buffer = queue.Queue()
    # 使用自定义的回调处理器
    callbacks = [CustomStreamingCallback(streaming_buffer)]
    return ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key, streaming=streaming,
                      callbacks=callbacks)



def process_streaming_output(streaming_buffer):
    """
    在独立线程中处理流式输出的函数
    """
    while True:
        token = streaming_buffer.get()
        if token is None:  # 检查结束信号
            logger.info("转录器接收到结束信号")
            shared_output.put(token)
            break
        time.sleep(0.05)
        shared_output.put(token)  # 将处理后的输出放入共享队列中
        logger.info(f"转录进行中: {token}")

def get_response_from_model(chat_instance, system_msg):
    # 创建一个队列来存放模型的流式输出
    streaming_buffer = queue.Queue()
    thread = threading.Thread(target=process_streaming_output, args=(streaming_buffer,))
    thread.start()
    # 使用自定义的回调处理器
    callbacks = [CustomStreamingCallback(streaming_buffer)]
    # 传递回调处理器到模型中
    chat_instance.callbacks = callbacks
    # 开始请求模型并获取响应
    chat_instance(messages=system_msg)
    # 使用循环从队列中获取模型的流式输出
    #while True:
        #token = streaming_buffer.get()
        #if token is None:  # 检查结束信号
           # break
        #time.sleep(0.05)
       # yield token
    # 启动处理流式输出的线程

# 界面模块可以从shared_output队列中读取数据并显示




