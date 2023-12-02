# -*- encoding: utf-8 -*-
"""
@author: acedar  
@time: 2023/8/20 12:33
@file: csv_load_verify.py 
"""


from langchain.document_loaders import CSVLoader, UnstructuredCSVLoader

loader = CSVLoader(file_path="../datasets/检索数据demo.csv", encoding='utf-8')
docs = loader.load()
print(docs[0])

loader = UnstructuredCSVLoader(file_path="../datasets/检索数据demo.csv", encoding='utf-8')
docs = loader.load()
print(docs[0])
