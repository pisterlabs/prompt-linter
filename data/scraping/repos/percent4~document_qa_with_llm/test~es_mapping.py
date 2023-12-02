# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: es_mapping.py
# @time: 2023/7/22 11:38
from elasticsearch import Elasticsearch

# 连接Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# 创建新的index
# mapping = {
#     'properties': {
#         'source': {
#             'type': 'text'
#         },
#         'cont_id': {
#             'type': 'integer'
#         },
#         'content': {
#             'type': 'text',
#             'analyzer': 'ik_max_word',
#             'search_analyzer': 'ik_smart'
#         }
#     }
# }
#
# es.indices.create(index='docs', ignore=400)
# result = es.indices.put_mapping(index='docs', body=mapping)
# print(result)

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 指定要使用的文档加载器
documents = TextLoader('../files/dengyue.txt', encoding='utf-8').load()
# 接下来，我们将文档拆分成块。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
datas = []
for i, text in enumerate(texts):
    source = text.metadata['source']
    content = text.page_content
    print(i + 1, source, content)
    datas.append({"source": source, "cont_id": i + 1, "content": content})

for data in datas:
    es.index(index='docs', document=data)



# 插入新的数据
# datas = [
#     {
#         'content': '美国留给伊拉克的是个烂摊子吗',
#         'source': 'http://view.news.qq.com/zt2011/usa_iraq/index.htm'
#     },
#     {
#         'content': '公安部：各地校车将享最高路权',
#         'source': 'http://www.chinanews.com/gn/2011/12-16/3536077.shtml'
#     }
# ]
#
# for data in datas:
#     es.index(index='docs', document=data)
