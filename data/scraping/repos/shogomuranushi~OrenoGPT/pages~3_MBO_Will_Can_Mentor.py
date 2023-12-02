import streamlit as st
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

# 定義
title = "目標管理 Will/Can 作成支援メンター。気付きを与えてくれます"
system_prompt = """あなたは優秀なメンターです。人事制度の目標管理、MBOのWill/Canを設定したいので、キーワードをもとに文章を校正してください。"""
user_prompt_template = """キーワードを参考に{sub_title}の文章をビジネス文章として分量を増やし校正してください。
ポイントとしては「なぜそれをするべきか、しない場合は何が損失になるか、それを実現するために具体的なアクションは」というのをQCDの観点でステップバイステップで考えて校正してください。
最後にサマリーとしてまとめてください。
【キーワード】
{keyword}
【生成文】
"""
# チャットボットとやりとりする関数
def api_call(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    return response

def create_exercise(keyword, sub_title):
    user_prompt = user_prompt_template.format(keyword=keyword, sub_title=sub_title)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = api_call(messages)
    return response

# UIの構築
st.title(title)

tab1, tab2, tab3, tab4 = st.tabs(["Will_1", "Will_2", "Can_1", "プロンプト"])

with tab1:
    sub_title = "今の仕事においてあなたが主体者として実現したいこと"
    keyword = st.text_input(sub_title)
    if st.button("生成", key='button1'):
        response = create_exercise(keyword, sub_title)
        with st.empty():
            exercise = ""
            for chunk in response:
                tmp_exercise = chunk["choices"][0]["delta"].get("content", "")
                exercise += tmp_exercise
                st.write("💻: " + exercise)

with tab2:
    sub_title = "2-3年後のキャリアビジョン"
    keyword = st.text_input(sub_title)
    if st.button("生成", key='button2'):
        response = create_exercise(keyword, sub_title)
        with st.empty():
            exercise = ""
            for chunk in response:
                tmp_exercise = chunk["choices"][0]["delta"].get("content", "")
                exercise += tmp_exercise
                st.write("💻: " + exercise)
with tab3:
    sub_title = "活かしたい強み・提供可能な価値"
    keyword = st.text_input(sub_title)
    if st.button("生成", key='button3'):
        response = create_exercise(keyword, sub_title)
        with st.empty():
            exercise = ""
            for chunk in response:
                tmp_exercise = chunk["choices"][0]["delta"].get("content", "")
                exercise += tmp_exercise
                st.write("💻: " + exercise)

with tab4:
    st.text_area("システムプロンプト", system_prompt)
    st.text_area("ユーザプロンプト", user_prompt_template)
