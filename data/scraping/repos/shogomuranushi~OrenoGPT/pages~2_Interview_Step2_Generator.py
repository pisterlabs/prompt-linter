import streamlit as st
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

# 定義
title = "二次面接質問ジェネレーター"
system_prompt = """あなたは優秀な二次面接担当者です。二次面接を成功させるための質問を生成してください。ステップバイステップで考えましょう。
"""
user_prompt_template = """キーワードに入っている内容は一次面接官が確認出来なかった項目です。
一次面接官が確認出来なかったその疑問点を解消したいです。解消しないと採用した後に問題が生じる可能性があります。
必ず解消させておきたいので解消するための質問文を生成してください。
直接聞いてしまうと答えが誘導されてしまうので、直接的な質問ではなく間接的な迂回した質問を生成してください。

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
    keyword = st.text_area("キーワードを入れると、その内容を踏まえて生成されます。一つの問いずつ入れると精度が上がりやすいです。")
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
