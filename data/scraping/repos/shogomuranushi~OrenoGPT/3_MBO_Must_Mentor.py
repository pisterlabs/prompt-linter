import streamlit as st
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

# 定義
title = "目標管理 Must 作成支援メンター。気付きを与えてくれます"
system_prompt = """あなたは優秀なメンターです。人事制度の目標管理、MBOのMust（何を目標にするか）を設定したいので、キーワードをもとに目標を立てて校正してください。
さらに目標に対して定量的なアクションを設定してください。目標は6ヶ月で達成できる内容にしてください。
"""
user_prompt_template = """キーワードを参考に{sub_title}の文章をビジネス文章として分量を増やし校正してください。
ステップバイステップで考えましょう。
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

tab1, tab2 = st.tabs(["Must", "プロンプト"])

with tab1:
    sub_title = "目標設定"
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
    st.text_area("システムプロンプト", system_prompt)
    st.text_area("ユーザプロンプト", user_prompt_template)
