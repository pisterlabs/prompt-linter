#!/usr/bin/env python3

# LangChain调用自定义语言模型：
# 创建一个LLM的衍生类，实现 _call 方法（用于接收输入字符串并返回响应字符串）以及一个可选方法：_identifying_params（用于帮助打印此类的属性）。

# 先从以下链接下载 llama-2-7b-chat.ggmlv3.q4_K_S.bin 模型，这是 TheBloke 使用 Llama 微调后的模型。
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

# 为了使用 llama-2-7b-chat.ggmlv3.q4_K_S.bin 这个模型，需要安装 pip3 install llama-cpp-python==0.1.78 这个包

from llama_cpp import Llama
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM

MODEL_NAME = "llama-2-7b-chat.ggmlv3.q4_K_S.bin"
MODEL_PATH = "/Users/xiaojunhuang/Documents/my/code/LLM/"

# 自定义LLM类，继承自基类LLM
class CustomLLM(LLM):
	model_name = MODEL_NAME

	def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
		prompt_length = len(prompt) + 5
		llm = Llama(model_path = MODEL_PATH + MODEL_NAME, n_threads = 4)
		response = llm(f"Q: {prompt} A: ", max_tokens = 256)
		output = response["choices"][0]["text"].replace("A: ", "").strip()
		# 返回生成的回复，同时剔除了问题部分和额外字符
		return output[prompt_length:]

	@property
	def _identifying_params(self) -> Mapping[str, Any]:
		return { "name_of_model": self.model_name }

	@property
	def _llm_type(self) -> str:
		return "custom"

llm = CustomLLM()
result = llm("昨天有一个客户抱怨他买了花给女朋友之后，两天花就枯了，你说作为客服我应该怎么解释？")
print(result)
# 运行11分钟左右得到结果：
# "Thank you for reaching out to us about your concern regarding the freshness of the flowers you purchased. We understand that it's important to have a pleasant experience with our products, and we apologize for any inconvenience this has caused.
# Firstly, we would like to clarify that the freshness of our flowers is guaranteed for a period of 3 days from the delivery date. Unfortunately, it seems that the flowers were delivered to you on [date] and the guarantee period has already passed. However, we want to assure you that we take any issue regarding the quality of our products very seriously, and we will do our best to address your concern.
# We would like to offer you a replacement bouquet or a full refund, whichever you prefer. Please let us know which option you would like us to proceed with, and we will arrange for it as soon as possible. We value your feedback and appreciate your patience in this matter."
	