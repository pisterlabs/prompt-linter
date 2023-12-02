# -*- coding:utf-8 -*-
# @FileName  : files_parser.py
# @Time      : 2023/8/8
# @Author    : LaiJiahao
# @Desc      : 加载文档

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FileParser:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20
        )

    def load(self, filename: str, file_type: str):
        """
        加载文档
        :param filename: 文件名称
        :param file_type: 文件格式
        :return:
        """
        file_path = f"data/{filename}.{file_type}"
        if file_type == "csv":
            return self._read_csv(file_path)
        if file_type == "docs":
            return self._read_docs(file_path)
        if file_type == "pdf":
            return self._read_pdf(file_path)
        if file_type == "md":
            return self._read_markdown(file_path)

    def _read_markdown(self, file_path: str):
        loader = UnstructuredMarkdownLoader(file_path=file_path)
        data = loader.load()
        data = self.text_splitter.split_documents(data)
        return data

    def _read_docs(self, file_path: str):
        loader = UnstructuredWordDocumentLoader(file_path=file_path)
        data = loader.load()
        data = self.text_splitter.split_documents(data)
        return data

    def _read_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path=file_path)
        data = loader.load()
        data = self.text_splitter.split_documents(data)
        return data

    def _read_csv(self, file_path: str):
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
        data = loader.load()
        return data
