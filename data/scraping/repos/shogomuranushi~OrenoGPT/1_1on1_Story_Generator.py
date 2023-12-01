import streamlit as st
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

# 定義
title = "1on1ネタジェネレーター"
system_prompt = """あなたは優秀なメンターです。
1on1を成功させるために1on1のネタの作成を行います。ステップバイステップで考えましょう。
"""
user_prompt_template = """キーワードを参考に1on1のネタを作成してください。
【キーワード】
{keyword}
【1on1ネタ】
"""

# チャットボットとやりとりする関数
def api_call(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    return response

def create_exercise(keyword):
    user_prompt = user_prompt_template.format(keyword=keyword)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = api_call(messages)
    return response

# UIの構築
st.title(title)

tab1, tab2= st.tabs(["生成", "プロンプト"])

with tab1:
    keyword = st.text_input("キーワードを入れると、その内容を踏まえて生成されます。")
    if st.button("生成"):
        response = create_exercise(keyword)
        with st.empty():
            exercise = ""
            for chunk in response:
                tmp_exercise = chunk["choices"][0]["delta"].get("content", "")
                exercise += tmp_exercise
                st.write("💻: " + exercise)

with tab2:
    st.text_area("システムプロンプト", system_prompt)
    st.text_area("ユーザプロンプト", user_prompt_template)
