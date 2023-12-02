# #!./.conda/envs/py310/bin/python3.10

# This is the file of using OpenAI APIs to analyze financial documents, see:https://github.com/openai/openai-cookbook/blob/main/examples/third_party_examples/financial_document_analysis_with_llamaindex.ipynb
# author: Xinyi Li
# contact: xl4412@nyu.edu
# time of completion: 2023-08-10

# !pip install llama-index pypdf
import openai
import json
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index import set_global_service_context
from llama_index.response.pprint_utils import pprint_response
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from llama_index.prompts import Prompt

from llama_index.schema import Node
from llama_index.response_synthesizers import get_response_synthesizer
import nltk
nltk.download('punkt')
# Error:
#   For more information see: https://www.nltk.org/data.html
#   Attempted to load tokenizers/punkt/english.pickle
#   Searched in:
#     - 'C:\\Users\\user/nltk_data'
#     - 'C:\\Users\\user\\Desktop\\SignalA_up\\AItag\\nltk_data'
#     - 'C:\\Users\\user\\Desktop\\SignalA_up\\AItag\\share\\nltk_data'
#     - 'C:\\Users\\user\\Desktop\\SignalA_up\\AItag\\lib\\nltk_data'
#     - 'C:\\Users\\user\\AppData\\Roaming\\nltk_data'
#     - 'C:\\nltk_data'
#     - 'D:\\nltk_data'
#     - 'E:\\nltk_data'
#     - 'C:\\Users\\user\\AppData\\Local\\llama_index'
#     - ''
# **********************************************************************
# nltk.download('punkt')
# [nltk_data] Error loading punkt: Remote end closed connection without
# [nltk_data]     response
# False

# Resolved:
# 手动下载punkt到上方任意目录即可，下载链接：https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip

from proj import *
from 文章 import *

# def get_embedding(text, model='text-embedding-ada-002'):
#     text = text.replace("\n"," ")
#     return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
# #
# #
# # dfa = pd.read_csv('data/report_clean.csv')
# # dfa = dfa[['TradingDate','InfoTitle','Detail','BulletinType','InnerCode','SecuCode']].reset_index(drop=True)
# # dfa = dfa.sort_values("TradingDate").reset_index(drop=True)
# #
# # group = dfa.groupby('InnerCode')
# # top_n = 10
# # # 保留最近的10条做测试
# # df_test = group.tail(top_n)
# # # 使用训练集的最后50条分类
# # df = dfa.drop(df_test.index).groupby('InnerCode').tail(top_n*5)
# # df = df.dropna()
# # df["combined"] = (
# #     df.InfoTitle.str.strip() + " " + df.Detail.str.strip()
# # )
# # # df.drop("Time", axis=1, inplace=True)
# # embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# # max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
# #
# # encoding = tiktoken.get_encoding(embedding_encoding)
# #
# # # omit reviews that are too long to embed
# # df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
# # df = df[df.n_tokens <= max_tokens]
# # len(df)
# # # df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model)
#
#
#
# prompt = "在分红公告，股东大会决议公告，业绩预告公告，持股变动公告，资产重组公告，再融资公告，股权激励公告，关联交易公告，担保公告，退市风险公告，交易所交易公开信息公告，现金管理公告，会计政策变更公告，人员聘请公告，审计保留意见公告，IPO公告，变更信息公告，内部控制公告，新项目开展公告这19个标签中，给出下面公告的最可能标签："
# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "user", "content": prompt.strip()},
#   ]
# )

def load_api_key(secrets_file):
    with open(secrets_file) as f:
        secrets = json.load(f)
    return secrets["OPENAI_API_KEY"]

# secretS.json is the file for your api keys you created
api_key = load_api_key("proj/secretS.json")
openai.api_key = api_key

model_name = 'gpt-3.5-turbo' # change to the model you need, available models see:https://platform.openai.com/docs/models
file_path = './your_report.pdf'

# set the gloabl default model to use
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",openai_api_key=api_key)
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context=service_context)

# upload and index your file
# PDF docs are converted to plain text `Document` objects, it takes time to operate
doc = SimpleDirectoryReader(input_files=[file_path]).load_data()
index = VectorStoreIndex.from_documents(doc)


# Simple Q&A
fin_engine = index.as_query_engine(similarity_top_k=3)
response = await fin_engine.aquery("your_question")
print(response)

#advanced Q&A


chat_text_qa_msgs = [
    HumanMessagePromptTemplate.from_template(
        "填写公告的分类和组成的表格.\n| 公告类型 | 公告内容 |"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
    ),
]
chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
text_qa_template = Prompt.from_langchain_prompt(chat_text_qa_msgs_lc)

# Refine Prompt
chat_refine_msgs = [
    SystemMessagePromptTemplate.from_template(
        "Always answer the question, even if the context isn't helpful."
    ),
    HumanMessagePromptTemplate.from_template(
        "We have the opportunity to refine the original answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "Original Answer: {existing_answer}"
    ),
]

chat_refine_msgs_lc = ChatPromptTemplate.from_messages(chat_refine_msgs)
refine_template = Prompt.from_langchain_prompt(chat_refine_msgs_lc)
