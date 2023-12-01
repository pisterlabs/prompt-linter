import os
import getpass
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import requests
import json

# os.environ['OPENAI_API_KEY'] = "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"
raw_documents = TextLoader('../quest2.txt', encoding='utf8').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

#documents转换为list
documents = list(documents)

content_list = documents[0].page_content.split('\n')
# print(content_list)

def requestChatGlm(str):
    url = 'http://176.10.10.229:8000'
    data = {
        "history": [],
        "prompt": str,
    }
    response = requests.post(url, json=data)
    response_data = response.json()
    # 处理响应数据
    print(response_data)


#遍历 content_list
for i in range(len(content_list)):
    prompts = "请模仿以下提示生成3条新的提示："+content_list[i]
    requestChatGlm(prompts)





