# 首先实现基本配置
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredFileLoader  #MP4视频用

#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import glob
#from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline

# 使用前配置自己的 api 到环境变量中如
import os
class DocumentLoader:
    def __init__(self):
        self.docs = []

    def pdf(self, filename):
        flist = glob.glob(filename)
        counts = len(flist)
        loaders = []
        for count in range(0,counts):
            loaders.append(flist[count])
        for one_loaders in loaders:
            loader = PyMuPDFLoader(one_loaders)
            self.docs.extend(loader.load())
        return self.docs

    def md(self, filename):
        files = glob.glob(filename)
        loaders = []
        for one_file in files:
            loader = UnstructuredMarkdownLoader(one_file)
            loaders.append(loader)
        for loader in loaders:
            self.docs.extend(loader.load())
        return self.docs

    def txt(self, filename):
        flist = glob.glob(filename)
        counts = len(flist)
        loaders = []
        for count in range(0,counts):
            loaders.append(flist[count])
        for loader in loaders:
            self.docs.extend(loader.load())
        return self.docs

    def split(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)
        return split_docs


