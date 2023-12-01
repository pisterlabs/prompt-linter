#!/usr/bin/env python3
# -*- coding utf-8 -*-
import yaml
import torch
import transformers
from transformers import AutoTokenizer
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain

'''
用于演示如何将HuggingFace Pipeline和LangChain集成
'''

def get_hf_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        api_key = yaml_data["huggingface"]["api_key"]
        return api_key

if __name__ == '__main__':
    # 导入HuggingFace API Token
    get_hf_key()

    # 指定预训练模型的名称
    model = "meta-llama/Llama-2-7b-chat-hf"

    # 从预训练模型中加载词汇器
    tokenizer = AutoTokenizer.from_pretrained(model)

    # 创建一个文本生成的管道
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        max_length = 1000
    )

    # 创建HuggingFacePipeline实例
    llm = HuggingFacePipeline(pipeline = pipeline, 
            model_kwargs = {'temperature':0})

    # 定义输入模板，该模板用于生成花束的描述
    template = """
               为以下的花束生成一个详细且吸引人的描述：
               花束的详细信息：
               ```{flower_details}```
               """

    # 使用模板创建提示
    prompt = PromptTemplate(template=template, 
                input_variables=["flower_details"])

    # 创建LLMChain实例
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # 需要生成描述的花束的详细信息
    flower_details = "12支红玫瑰，搭配白色满天星和绿叶，包装在浪漫的红色纸中。"

    # 打印生成的花束描述
    print(llm_chain.run(flower_details))
