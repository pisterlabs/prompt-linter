#ref:https://qiita.com/tsuru_/items/049097bc51c974fc291d
import streamlit as st
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

#libsフォルダの中にあるaoai_test.pyをimport
from libs.aoai_test import generate_response_aoai
from libs.exsample_selector import *

#タイトルを表示
st.title('🦜FAQ回答')

st.markdown("""

## 概要
ユーザーの質問内容から意味検索を行い、最も意味的に近いQを探し出し、
それに対応するAを回答します。
意味的な検索のため、従来よりもより柔軟な検索ができます

例
「私の家でもなんとかなる？」という質問を投げかけると
もっとも意味の近い
「Q: 新型コロナウイルス感染症に罹った場合、自宅療養は可能ですか？」というQがヒットし
そのAを返します

## デモ用エクセル
以下はデモ用に作成したコロナウイルス感染症FAQです。以下の表を直接操作して
内容を追加/編集/削除することも可能です。
""")

df = pd.read_excel('gptdemo/data/qa/virusQA.xlsx',sheet_name='シート1',
        index_col='QAの番号')

# if st.button('デモ用FAQを表示する'):
#     # df = pd.read_excel('gptdemo/data/qa/virusQA.xlsx',sheet_name='シート1',
#     #     index_col='QAの番号')
#     st.table(df)

edited_df = st.data_editor(df,num_rows="dynamic")

#最後に、st.form() を使用して、ユーザーが入力したプロンプトを受け入れるためのテキストボックス (st.text_area()) を作成します。
#ユーザーが「送信」ボタンをクリックした時点で、prompt 入力変数 (テキスト) を引数として、generate-response() 関数が呼び出されます。
k = st.selectbox(label="回答数を選択",
             options=[1,2,3,4,5,6,7,8,9,10])


with st.form('my_form'):
  text = st.text_area('質問したい内容を入力:', '')
  submitted = st.form_submit_button('Submit')
  if submitted:
    # answer = get_qa(text,k)
    answer = get_qa_edited_df(text,k,edited_df)
    st.info(answer)