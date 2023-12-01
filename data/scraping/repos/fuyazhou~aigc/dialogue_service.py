#!/usr/bin/env python
# -*- coding:utf-8 _*-

import os
import sys
import openai
import re
from duckduckgo_search import ddg
from langchain.document_loaders import UnstructuredFileLoader
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from logging import getLogger
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
import config
import prompt_template
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

os.environ["OPENAI_API_KEY"] = config.openai_api_key
os.environ["SERPAPI_API_KEY"] = config.SERPAPI_API_KEY

embedding = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Dialogue_Service:
    def __init__(self):
        self.character_dialogue_precision_qa_chain = None
        self.db = None
        self.docs_path = config.docs_path
        self.db_path = config.vector_store_path

    def init_source_vector(self):
        """
            初始化本地知识库向量
            :return:
        """
        logger.info("******* init_source_vector ******")
        self.db = FAISS.from_documents(self.load_data(self.docs_path), embedding)
        self.db.save_local(self.db_path)
        logger.info("******* init_source_vector done ******")

    def load_data(self, data_path):
        """
            加载数据
            :return:
        """

        logger.info("*******start load data*********")
        docs = []
        if os.path.isdir(data_path):
            for sub_data_path in os.listdir(data_path):
                if sub_data_path.endswith('.txt'):
                    print(sub_data_path)
                    loader = UnstructuredFileLoader(f'{data_path}/{sub_data_path}')
                    doc = loader.load()

                    all_page_text = [p.page_content for p in doc]
                    joined_page_text = "\n".join(all_page_text)
                    docs.append(joined_page_text)

                if sub_data_path.endswith('.docx'):
                    print(sub_data_path)
                    loader = Docx2txtLoader(f'{data_path}/{sub_data_path}')
                    doc = loader.load()

                    all_page_text = [p.page_content for p in doc]
                    joined_page_text = "\n".join(all_page_text)
                    docs.append(joined_page_text)

                if sub_data_path.endswith('.pdf'):
                    print(sub_data_path)
                    loader = PyPDFLoader(f'{data_path}/{sub_data_path}')
                    doc = loader.load()

                    all_page_text = [p.page_content for p in doc]
                    joined_page_text = "\n".join(all_page_text)
                    docs.append(joined_page_text)

        if os.path.isfile(data_path):
            if data_path.endswith('.txt'):
                print(data_path)
                loader = UnstructuredFileLoader(data_path)
                doc = loader.load()

                all_page_text = [p.page_content for p in doc]
                joined_page_text = "\n".join(all_page_text)
                docs.append(joined_page_text)

            if data_path.endswith('.docx'):
                print(data_path)
                loader = Docx2txtLoader(data_path)
                doc = loader.load()
                all_page_text = [p.page_content for p in doc]
                joined_page_text = "\n".join(all_page_text)
                docs.append(joined_page_text)
            if data_path.endswith('.pdf'):
                print(data_path)
                loader = PyPDFLoader(data_path)
                doc = loader.load()
                all_page_text = [p.page_content for p in doc]
                joined_page_text = "\n".join(all_page_text)
                docs.append(joined_page_text)
        logger.info("*******load data done********")
        docs_more = []
        docs_more.extend(re.split("第[一二三四五六七八九十]{0,3}[条]", docs[0]))
        docs_more.extend(re.split("第[一二三四五六七八九十]{0,3}[条]", docs[1]))
        docs_more.extend(re.split("第[一二三四五六七八九十]{0,3}[章]", docs[2]))
        docs = docs_more
        logger.info(f"total have {len(docs)} datas")
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=700, separator="\n")
        documents = text_splitter.create_documents(docs)
        return documents

    def add_document(self, document_path):
        logger.info("******** start add_document ***********")
        self.db.add_documents(self.load_data(document_path))
        self.db.save_local(self.db_path)
        logger.info("******** start done ***********")

    def init_character_dialogue_precision_qa_chain(self):
        logger.info("********** init  character_dialogue_precision_qa_chain**********")
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template.template3)
        # Run chain
        self.character_dialogue_precision_qa_chain = RetrievalQA.from_chain_type(
            llm,
            # retriever=self.db.as_retriever(search_type="mmr"),
            retriever=self.db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    # def load_vector_store(self, path):
    #     if path is None:
    #         self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
    #     else:
    #         self.vector_store = FAISS.load_local(path, self.embeddings)
    #     return self.vector_store

    def character_dialogue_precision_qa(self, query):
        """
        角色对话-精准：
            【精准模式】中提问只返回本地知识库的数据，如未搜索到则返回暂未学习。
            例：问题；残疾人。答案：残疾人的介绍、残疾人的相关政策、申请残疾人证的办理流程等。
            :return:
        """
        try:
            result = self.character_dialogue_precision_qa_chain(query)
            result = result["result"]
            return result
        except Exception as e:
            logger.info(f"something wrong character_dialogue_precision:{e}")
            return config.common_responses


if __name__ == '__main__':
    dialogue_service = Dialogue_Service()
    dialogue_service.init_source_vector()
    dialogue_service.init_character_dialogue_precision_qa_chain()
    query = "你好"
    result = dialogue_service.character_dialogue_precision_qa(query)
    print(result)

    query = "人才补贴"
    result = dialogue_service.character_dialogue_precision_qa(query)
    print(result)

    query = "残疾人"
    result = dialogue_service.character_dialogue_precision_qa(query)
    print(result)
