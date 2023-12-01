import os
import gradio as gr
import random
import torch
# import cv2
import re
# import uuid
# from PIL import Image
import numpy as np
import argparse

from langchain.agents import AgentType

from nft_tools import NFT_info

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI

import openai
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
tools = []
# 导入的类
# models = {"NFT_info":NFT_info}
class_name = 'NFT_info'
device = 'cpu'
models = {}
models[class_name] = globals()[class_name](device=device)

for instance in models.values():
    # 获取类中所有的属性和方法名称
    for e in dir(instance):
        print(e)
        if e.startswith('get_nft'):
            # getattr(obj, name)时，Python会尝试获取obj对象的name属性或方法。
            func = getattr(instance, e)
            tools.append(Tool(name=func.name, description=func.description, func=func))
print(tools)
mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
#
# mrkl.run("有哪些关于Bored Ape Yacht Club这collection_address的地址是什么？")
mrkl.run("有哪些关于Bored Ape Yacht Club这collection_address的地址是什么？")
# if __name__ == '__main__':

    # print(get_nft_by_contract('0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d','4495'))