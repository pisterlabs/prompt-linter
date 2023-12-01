import os
import openai

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPEN_AI_KEY")

#ChatGPTによる文章生成
def create_auto_article(user_input_target,user_input_target_ploblem,
                          user_input_emotion,user_input_ploblem_reason,
                          user_input_authority,user_input_goods,user_input_appeal,
                          user_input_review,user_input_price,user_input_special_price,
                          user_input_example_title,user_input_example_text,
                          user_target_design,text_position_design,
                          speech_bubble_color_design,large_heading_color_design,font_design,
                          user_input__SEO_keyword):

    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
    {"role": "system", "content": "あなたはブログ記事を生成する人です。\
      すべてHTML形式で出力してください。デザインを施してください。\
     また、SEO対策をしてください。SEO対策として、SEOキーワードを多く盛り込んでください。\
     ブログとして、十分な量の記事を生成してください。"},
    {"role": "user", "content": f"あなたは以下のデータを基に記事を作成します。\
     また、すべてHTML形式で出力してください。\
     1000字から2000字程度で記事を生成してください\
     ターゲットは{user_input_target}\
    ターゲットの問題は{user_input_target_ploblem}\
    悩みから生まれる感情は{user_input_emotion}\
    悩みのの原因は{user_input_ploblem_reason}\
    権威は{user_input_authority}\
    商品名は{user_input_goods}\
    アピール,訴求は{user_input_appeal}\
    レビューは{user_input_review}\
    価格は{user_input_price}\
    特別価格は{user_input_special_price}\
    参考にする記事の見出しは{user_input_example_title}\
    参考にする記事の本文は{user_input_example_text}\
    以降はデザインをする際に参考にするデータです。\
    ターゲットは{user_target_design}\
    テキストの位置は{text_position_design}\
    吹き出しの色は{speech_bubble_color_design}\
    見出し(大)の色は{large_heading_color_design}\
    フォントは{font_design}\
    SEOキーワードは{user_input__SEO_keyword}"},
    ]
    )
    return  response.choices[0].message.content

# Streamlitのタイトルを設定
st.title('記事自動生成')

# ユーザー入力用のテキストボックスを設置
user_input__SEO_keyword = st.text_area("SEOキーワード")
user_input_target = st.text_area("ターゲット")
user_input_target_ploblem= st.text_area("ターゲットの問題")
user_input_emotion= st.text_area("悩みから生まれる感情")
user_input_ploblem_reason= st.text_area("悩みの原因")
user_input_authority= st.text_area("権威")
user_input_goods= st.text_area("商品名")
user_input_appeal= st.text_area("アピール,訴求")
user_input_review= st.text_area("口コミ")
user_input_price= st.text_area("価格")
user_input_special_price= st.text_area("特別価格")
user_input_example_title=st.text_area("参考にしたい記事の見出し")
user_input_example_text=st.text_area("参考にしたい記事の本文")
#カスタムインプットを作りたいなら事前に何個か用意するのが良さそう

user_target_design=st.radio("ターゲット", ("若い女性", "大人な女性", "若い男性","大人な男性"), horizontal=True)
text_position_design=st.radio("テキストの位置", ("中央寄せ", "左寄せ"), horizontal=True)
speech_bubble_color_design=st.radio("吹き出しの色", ("グレー", "青","オレンジ"), horizontal=True)
large_heading_color_design=st.radio("見出し(大)の色", ("グレー", "青","オレンジ"), horizontal=True)
font_design=st.radio("フォント", ("ゴシック", "明朝","ヒラギノ"), horizontal=True)


if st.button('生成開始'):
            
    result = create_auto_article(user_input_target,user_input_target_ploblem,
                          user_input_emotion,user_input_ploblem_reason,
                          user_input_authority,user_input_goods,user_input_appeal,
                          user_input_review,user_input_price,user_input_special_price,
                          user_input_example_title,user_input_example_text,
                          user_target_design,text_position_design,
                          speech_bubble_color_design,large_heading_color_design,font_design,
                          user_input__SEO_keyword)
    
    st.write("生成記事:")
    st.markdown(result, unsafe_allow_html=True)