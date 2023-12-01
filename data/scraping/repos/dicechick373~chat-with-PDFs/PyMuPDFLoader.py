
from langchain.document_loaders import PyMuPDFLoader

def PDF_PyMuPDFLoader(pdf_path):
    """
    Load a PDF file using PyMuPDFLoader and split it into data.

    Parameters:
        pdf_path (str) : The path to the PDF file to be loaded.

    Returns:
        data (List[Document]) : The loaded and split data.
    """
    loader = PyMuPDFLoader(pdf_path)
    data= loader.load_and_split()

    return data



# '''
# 表を整形したいので、ChartGPTに変換してもらう
# '''
# from langchain.llms import OpenAIChat
# import streamlit as st
# import os


# # OpenAIを利用するためにプロキシ設定が必要
# os.environ['http_proxy'] = st.secrets["proxy"]["URL"]
# os.environ['https_proxy'] = st.secrets["proxy"]["URL"]


# llm = OpenAIChat(openai_api_key=st.secrets["api_keys"]["OPEN_API_KEY"],temperature=0.0)

# prompt = f'''
# 以下に示すデータは、技術基準として提供されているPDFをUnstructuredPDFLoaderによりテキスト化したものである。
# 文章と表に分けて、JSONデータに変換せよ。


# # データ
# {data[0].page_content}
# '''
# print(prompt)
# result = llm(prompt)
# print(result)
