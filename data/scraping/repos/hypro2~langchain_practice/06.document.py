import os

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

import pandas as pd
from langchain.document_loaders import PyPDFLoader, DataFrameLoader, BSHTMLLoader
from langchain.schema import Document

"""
문서를 만들 수 있습니다. 
직접 document를 이용해서 문서를 만들 수 있고, 혹은 다른 타입의 파일을 불러와서 문서를 자동으로 생성 할 수 있습니다. 
이경우 앞에서 사용한 split을 같이 사용하게 됩니다.
이렇게 문서를 만들면 나중에 vector index에서 사용하기 용이해집니다.
"""

# document 만들기
def my_doc():
  my_page = Document(
    page_content="이 문서는 제 문서입니다. 다른 곳에서 수집한 텍스트로 가득합니다.",
    metadata={'explain': 'The LangChain Papers'})
  print(my_page)

def my_docs():
  my_list = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
  ]

  my_pages = [Document(page_content = i) for i in my_list]
  print(my_pages)


# PyPDF Loader # pip install pypdf
def pdf_doc():
  loader = PyPDFLoader("../dataset/2013101000021.pdf")
  pages = loader.load_and_split() # text_splitter를 입력을 수 있다.
  print(pages[:5])


# DataFrame Loader
def df_doc():
  df = pd.read_csv("../dataset/mlb_teams_2012.csv")
  loader = DataFrameLoader(df, page_content_column="Team")
  pages = loader.load_and_split()
  print(pages[:5])


# BS4 HTML Loader
def html_doc():
  loader = BSHTMLLoader("../dataset/fake-content.html")
  pages = loader.load_and_split()
  print(pages[:5])


if __name__=="__main__":
    my_doc()
    my_docs()
    pdf_doc()
    df_doc()
    html_doc()
