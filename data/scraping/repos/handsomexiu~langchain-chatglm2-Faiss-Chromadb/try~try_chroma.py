import sys
sys.path.append('../')
# 这里是为了能偶访问到非当前文件夹中的包

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.document_loaders import TextLoader#这个有一点问题
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import sentence_transformers
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from config import *
from typing import List
import re
import nltk

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))') 
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list
    
filepath="../knowledge/草稿.md"
# loader = UnstructuredFileLoader(filepath,mode="elements")
loader = UnstructuredFileLoader(filepath)
textsplitter = ChineseTextSplitter(pdf=False)
docs = loader.load_and_split(textsplitter)

print(docs)

embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
num_gpus = num_gpus#GPU数量
large_language_model = init_llm
embedding_model=init_embedding_model

model = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model], )
print('第一步加载成功')
model.client = sentence_transformers.SentenceTransformer(
            model.model_name,
            device=EMBEDDING_DEVICE,
            cache_folder=os.path.join(MODEL_CACHE_PATH,model.model_name))
print('embedding模型加载成功')
# 这里相当于是对client属性进行赋值，尽管在__init__huggingface中已经赋值了，但是没全
'''
        self.client = sentence_transformers.SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

'''
model.model_name

db = Chroma.from_documents(docs, model,persist_directory="../langchain_chromadb/vector_store/chroma_1")
print('chroma加载成功')