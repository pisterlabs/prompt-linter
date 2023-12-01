# -*- coding: utf-8 -*-
"""
Copyright (c) 2023, Nimblex Co .,Ltd.

@author: zhangwenping
"""
from pydoc import doc
from typing import List
from .kb_base import KBRepository
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from core.memfire_cloud import client as mf_client


# knowledge base repository
class MemFireKBRepository(KBRepository):
    def __init__(self, embedding=None):
        self.vector_store = None
        super().__init__(embedding)

    def search_docs(self, question: str) -> List[any]:
        # 从向量库中查询相似度高的文档并返回 matched_docs
        matched_docs = self.vector_store.similarity_search(question)
        return matched_docs

    def load_doc(self, doc_path: str) -> List[any]:
        # 处理文档：提取文本、切分、embedding、写入向量库
        raise NotImplementedError()

    def load_dir(self, dir_path: str) -> List[any]:
        # 加载本地目录
        loader = DirectoryLoader(dir_path, show_progress=True, silent_errors=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
        all_splits = text_splitter.split_documents(docs)

        self.vector_store = SupabaseVectorStore.from_documents(
            all_splits, self.embedding, client=mf_client)
