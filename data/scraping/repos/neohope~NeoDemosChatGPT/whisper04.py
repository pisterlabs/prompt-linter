#!/usr/bin/env python3
# -*- coding utf-8 -*-

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import SpacyTextSplitter
from llama_index import GPTListIndex, LLMPredictor, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

'''
总结文本
'''


if __name__ == '__main__':
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # 加载文本
    text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = 2048)
    parser = SimpleNodeParser(text_splitter=text_splitter)
    documents = SimpleDirectoryReader('./data/transcripts').load_data()
    nodes = parser.get_nodes_from_documents(documents)

    # 总结文本
    list_index = GPTListIndex(nodes=nodes, service_context=service_context)
    response = list_index.query("请你用中文总结一下我们的播客内容:", response_mode="tree_summarize")
    print(response)
