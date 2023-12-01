# coding=utf-8
from graia.ariadne.app import Ariadne
from graia.ariadne.event.message import GroupMessage
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.model import Group
import PIL.Image
from graia.saya import Channel
from PIL import ImageTk
import re
from graia.saya.builtins.broadcast.schema import ListenerSchema
import http.client
import hashlib
import urllib
import json
import openai
import os
import random
from tkinter import *
import tkinter as tk
from selenium import webdriver
from time import sleep
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import requests
import multiprocessing
import time
from multiprocessing import freeze_support
import subprocess
from graia.ariadne.event.mirai import NudgeEvent
from time import sleep
from typing import Union

import requests
import pyncm.apis.track
from urllib.parse import urlparse, parse_qs
import os

api_key = "sk-XXXXXXXXXXX"

openai.api_key = api_key

from datetime import datetime

from graia.ariadne.message.element import At, Plain, Image, Forward, ForwardNode

import textwrap
global text_box
global input_box
global text
global value
global message
global num
global struct
global a
global check_file_path
from PIL import Image



from PIL import ImageDraw, ImageFont
from graia.ariadne.message.chain import MessageChain
from graia.ariadne.message.element import Image as graia_Image

channel = Channel.current()

def os_open():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
def os_close():
    del os.environ['http_proxy']
    del os.environ['https_proxy']

def chatgpt(message):
    # 配置OpenAI API凭证


    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % openai.api_key
    }

    data = {
        "model": "gpt-3.5-turbo-16k",
        "messages": [{"role": "user", "content": "%s" % message}],
        "temperature": 0.5
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    return (response.json().get("choices")[0].get("message").get("content"))




def text_to_image(text, max_width=2000, max_height=2000,font_size = 100,font_path='simhei.ttf', text_color=(0, 0, 0), background_color=(255, 255, 255)):
    # 使用 PILImage 类别名代替原始的 Image 类
    image = Image.new('RGB', (max_width, max_height), background_color)
    draw = ImageDraw.Draw(image)

    # 逐渐减小字体大小，直到文字适应图像

    font = ImageFont.truetype(font_path, font_size)
    wrapped_text = textwrap.fill(text, width=int(max_width*0.8/font_size))
    text_width, text_height = draw.textsize(wrapped_text, font)
    while text_width > max_width or text_height > max_height:
        font_size -= 5
        font = ImageFont.truetype(font_path, font_size)
        wrapped_text = textwrap.fill(text, width=int(max_width*0.8/font_size))
        text_width, text_height = draw.textsize(wrapped_text, font)

    # 计算文字在图像中的位置
    x = (max_width - text_width) // 2
    y = (max_height - text_height) // 2

    # 将文字渲染到图像中心
    draw.text((x, y), wrapped_text, font=font, fill=text_color)

    # 保存图像为文件
    image.save("bot_content.png", format="PNG")


@channel.use(ListenerSchema(listening_events=[GroupMessage,NudgeEvent]))
async def ero(app: Ariadne, group: Group, message: MessageChain):

    if message.startswith("txtbot"):
        message = message.removeprefix("txtbot")
        result = message.display
        content = result
        print(content)
        os_open()
        result = chatgpt(content)
        os_close()
        await app.upload_file(data=result, target=group, name="%s.txt" % message)


    if message.startswith("bot"):
        message = message.removeprefix("bot")
        result = message.display
        content = result
        print(content)
        os_open()
        result = chatgpt(content)
        text_to_image(result)
        os_close()

        # 上传图像文件
        with open("bot_content.png", "rb") as image:
            await app.upload_file(data=image.read(), target=group, name="bot.png")
