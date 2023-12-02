'''
Author: leyi leyi@myun.info
Date: 2023-09-09 19:54:18
LastEditors: leyi leyi@myun.info
LastEditTime: 2023-09-09 20:24:32
FilePath: /learning-openai/sales_chatbot/sales_data.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

with open("real_sales_data.txt") as f:
    real_estate_sales = f.read()
    text_splitter = CharacterTextSplitter(
        separator=r'\d+\.',
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
    )
    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings(
        openai_api_key='sk-xxx'))
    db.save_local("real_sale_data")
