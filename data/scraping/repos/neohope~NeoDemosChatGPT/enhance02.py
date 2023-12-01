#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import SpacyTextSplitter
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

'''
llama_index对于文章进行小结
本例中，相当于对文本分段，按树状结构逐步向上进行总结

使用spaCy分词模型
python -m spacy download zh_core_web_sm
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    # 配置LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # 分词模型，最长2048个token
    text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = 2048)
    parser = SimpleNodeParser(text_splitter=text_splitter)

    # 加载目录中的语料
    documents = SimpleDirectoryReader('./data/mr_fujino').load_data()

    # 获取语料中的nodes
    nodes = parser.get_nodes_from_documents(documents)

    # 最简单的索引结构GPTListIndex
    list_index = GPTListIndex(nodes=nodes, service_context=service_context)

    # 树状模型进行总结
    response = list_index.query("下面鲁迅先生以第一人称‘我’写的内容，请你用中文总结一下:", response_mode="tree_summarize")
    print(response)
    
