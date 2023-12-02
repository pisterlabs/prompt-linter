import requests
import streamlit as st
import openai
import json
from streamlit_extras.stateful_button import button
import os

MONGO_URL_findOne = os.environ.get('MONGO_URL_findOne')
MONGO_URL_updateOne = os.environ.get('MONGO_URL_updateOne')
MONGO_URL_insertOne = os.environ.get('MONGO_URL_insertOne')
MONGO_KEY = os.environ.get('MONGO_KEY')

st.info("chatgpt for finance模型测试")


def openai_create(messages):
    openai.api_key = "EMPTY"  # Not support yet
    openai.api_base = "http://localhost:38080/v1"

    model = "vicuna-13b"

    # create a chat completion
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    # print the completion
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def insert_one(prompt, result1, result2, selected):
    payload = json.dumps({
        "collection": "vicuna_differ",
        "database": "vicuna-13b",
        "dataSource": "ChatgptUsing",
        "document": {"prompt": prompt,
                     "result1": result1,
                     "result2": result2,
                     "selected": selected}
    })
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': MONGO_KEY,
    }
    response = requests.request("POST", MONGO_URL_insertOne, headers=headers, data=payload)
    print(response.text)



st.title("chatgpt for finance打分")
input_words = st.text_area("请输入问题:", key="question_input")

prompt = [{"role": "user", "content": input_words}]

max_input_len = 2000

if st.button("确认", key="word_gpt3"):
    if len(input_words) < max_input_len:
        with st.spinner('答案生成中...'):
            result1 = openai_create(prompt)
            result2 = openai_create(prompt)
            st.balloons()
            st.success("大功告成！")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(result1)
                if button("选择答案 1", key="select_1"):
                    insert_one(input_words, result1, result2, "left")

            with col2:
                st.markdown(result2)
                if button("选择答案 2", key="select_2"):
                    insert_one(input_words, result1, result2, "right")



