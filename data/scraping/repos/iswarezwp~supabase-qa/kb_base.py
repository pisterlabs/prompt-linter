# -*- coding: utf-8 -*-
"""
Copyright (c) 2023, Nimblex Co .,Ltd.

@author: zhangwenping
"""
from operator import attrgetter
from os import getenv
from typing import List
from abc import ABC, abstractmethod

from langchain.embeddings import OpenAIEmbeddings
from core.helpers import get_qa_backend

# knowledge base repository
class KBRepository(ABC):
    def __init__(self, embedding=None):
        if embedding is None:
            embedding = self._get_embedding()
        self.embedding = embedding

    def _get_embedding(self):
        backend = get_qa_backend()
        if backend == 'openai':
            return self._get_openai_embedding()

        raise NotImplementedError(f"backend {backend} not supported")

    def _get_openai_embedding(self):
        return OpenAIEmbeddings(openai_api_key=getenv('OPENAI_API_KEY'))

    @abstractmethod
    def search_docs(self, question: str) -> List[any]:
        # 从向量库中查询相似度高的文档并返回 matched_docs
        raise NotImplementedError()

    @abstractmethod
    def load_doc(self, doc_path: str) -> List[any]:
        # 处理文档：提取文本、切分、embedding、写入向量库
        raise NotImplementedError()

    @abstractmethod
    def load_dir(self, dir_path: str) -> List[any]:
        # 加载本地目录
        raise NotImplementedError()
