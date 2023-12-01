#!/usr/bin/env python3

# HuggingFace 的 Pipline 简化了多种常见自然语言处理（NLP）任务使用流程，
# 使用户不用深入了解模型细节，也能容易利用预训练模型做任务。

model = "meta-llma/llama-2-7b-chat-hf"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

# 创建一个文本生成的管道
import transformers
import torch
pipeline = transformers.pipeline(
	"text-generation", # Transformers 支持多种NLP任务，包括但不限于文本生成("text-generation")、文本分类("text-classification")、问答("question-answering")、摘要("summarization")等
	model = model,
	torch_dtype = torch.float16, # 计算的数据类型设置为半精度浮点数，这样可以减少内存的使用，但可能牺牲一些数据的精度
	device_map = "auto",
	max_length = 1000 # 生成文本的最大长度
)
# 创建HuggingFacePipeline实例
from langchain import HuggingFacePipeline
llm = HuggingFacePipeline(
	pipeline = pipeline,
	model_kwargs = { "temperature": 0 }
)
# 使用模板创建提示
template = """
  为以下的花束生成一个详细且吸引人的描述：
  花束的详细信息：
  ```{flower_details}``
"""
from langchain import PromptTemplate, LLMChain
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt = prompt, llm = llm)

flower_details = "12支红玫瑰，搭配白色满天星和绿叶，包装在浪漫的红色纸中。"
print(llm_chain.run(flower_details))


