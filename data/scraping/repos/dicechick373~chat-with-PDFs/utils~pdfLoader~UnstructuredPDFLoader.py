

# 参考サイト
# https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.pdf.UnstructuredPDFLoader.html

'''
PDFをテキスト変換する関数
  dataはList[Document]
   page_content=''
   metadata={'source':""}
'''
from langchain.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("data/pdf/土木技術管理規程集_道路Ⅱ編_テスト1.pdf")
data= loader.load_and_split()


'''
表を整形したいので、ChartGPTに変換してもらう
'''
from langchain.llms import OpenAIChat
import streamlit as st
import os


# OpenAIを利用するためにプロキシ設定が必要
os.environ['http_proxy'] = st.secrets["proxy"]["URL"]
os.environ['https_proxy'] = st.secrets["proxy"]["URL"]


llm = OpenAIChat(openai_api_key=st.secrets["api_keys"]["OPEN_API_KEY"],temperature=0.0)

prompt = f'''
以下に示すデータは、技術基準として提供されているPDFをUnstructuredPDFLoaderによりテキスト化したものである。
文章と表に分けて、JSONデータに変換せよ。


# データ
{data[0].page_content}
'''
print(prompt)
result = llm(prompt)
print(result)
