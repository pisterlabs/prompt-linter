# -*- coding:utf-8 -*-
import os
from zipfile import ZipFile
import numpy as np
from lxml import etree
import xml.etree.ElementTree as ET
import zipfile
from docx import Document
import shutil
from langchain.llms.base import LLM
from typing import List, Optional
import requests
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader


class Vicuna(LLM):
    max_token: int = 2048
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024
    url_llm = "http://localhost:6007/llm"

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Vicuna"

    def llm(self, prompt: str):
        try:
            content1 = json.dumps({"text": prompt})
            response = requests.request("POST", self.url_llm, data=content1)
            res = response.content.decode('unicode_escape')
            return json.loads(res, strict=False)['response']
        except Exception as e:
            print(e)
            return "服务器已关闭，请联系服务器管理员"

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        response = self.llm(prompt)
        return response


def start():

    llm = Vicuna()
    embeddings = HuggingFaceEmbeddings(model_name='/root/report_qa/huggingface/infgrad/stella-large-zh')
    all_faiss_index={}
    count=1
    for i, dir_name in enumerate(os.listdir("./plain_text/")):
        # print(all_faiss_index)
        # print(i)
        # print(dir_name)
        temp={}
        for file_name in os.listdir("./plain_text/"+dir_name+"/"):
            # file_name=file_name.encode('UTF-8', 'ignore').decode('UTF-8')
            print("正在emmending的文件序号：",count)
            count+=1
            # whole_file_name = "./document/"+dir_name+"/" + file_name
            # loader = Docx2txtLoader(whole_file_name)
            #
            # data = loader.load()
            # # #
            path = "./plain_text/" +dir_name+"/"+ file_name[:-4] + ".txt"
            # # #
            # mode = 'w'
            # string = data[0].page_content
            #
            # with open(path, mode, encoding='utf-8') as f:
            #     # string = string.encode('utf-8')
            #     f.write(string)
            loader = TextLoader(path, autodetect_encoding=True)

            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            faiss_index = FAISS.from_documents(texts, embeddings)
            if dir_name=="钻井地质设计报告":
                temp[file_name.split('井')[0]+'井']=(faiss_index,file_name[:-4])
            elif dir_name=="油田开发年报":
                temp[file_name.split('年')[0]+'年'] = (faiss_index,file_name[:-4])
            elif dir_name == "气田开发年报":
                temp[file_name.split('年')[0] + '年'] = (faiss_index, file_name[:-4])

        all_faiss_index[dir_name]=temp

    print("all_faiss_index:", all_faiss_index)
    return all_faiss_index,llm



def start_table():
    
    embeddings = HuggingFaceEmbeddings(model_name='/root/report_qa/huggingface/infgrad/stella-large-zh')
    all_faiss_index_table={}
    count=1
    for i, dir_name in enumerate(os.listdir("./PDF/")):
        temp={} 
        for file_name in os.listdir("./PDF/"+dir_name+"/"):
            print("正在emmending的文件序号：",count)
            count+=1
            path = "./PDF/" +dir_name+"/"+ file_name[:-4] + ".pdf"
            loader = UnstructuredPDFLoader(path, mode="elements")
            documents = loader.load()

            faiss_index = FAISS.from_documents(documents, embeddings)
            if dir_name=="钻井地质设计报告":
                temp[file_name.split('井')[0]+'井']=(faiss_index,file_name[:-4])
            elif dir_name=="油田开发年报":
                temp[file_name.split('年')[0]+'年'] = (faiss_index,file_name[:-4])
            elif dir_name == "气田开发年报":
                temp[file_name.split('年')[0] + '年'] = (faiss_index, file_name[:-4])

        all_faiss_index_table[dir_name]=temp
        
    print("all_faiss_index_table:", all_faiss_index_table)
    return all_faiss_index_table

    


# all_faiss_index,llm=start()
# print(all_faiss_index)
# if __name__ == '__main__':
#     all_faiss_index,llm=start()
#     print(all_faiss_index)
