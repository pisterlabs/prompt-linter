import streamlit as st
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

# 定義
title = "セキュリティ定期試験問題ジェネレーター"
system_prompt = """あなたは優秀なセキュリティエンジニアです。
全社のセキュリティ向上のために、年に1回のセキュリティテストの練習問題の作成ととその回答、解説の作成を行います。ステップバイステップで考えましょう。
"""
user_prompt = """年に1回、全社員を対象としたセキュリティテストを行う必要があります。
初心者向けのセキュリティテストの問題を1問作成してください。正解と解説も加えてください。

【問題】

【正解】

【解説】
"""

# チャットボットとやりとりする関数
def api_call(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    return response

def create_exercise():
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
    if st.button("生成"):
        response = create_exercise()
        with st.empty():
            exercise = ""
            for chunk in response:
                tmp_exercise = chunk["choices"][0]["delta"].get("content", "")
                exercise += tmp_exercise
                st.write("💻: " + exercise)

with tab2:
    st.text_area("システムプロンプト", system_prompt)
    st.text_area("ユーザプロンプト", user_prompt)