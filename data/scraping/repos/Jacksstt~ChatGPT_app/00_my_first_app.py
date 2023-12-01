import streamlit as st
from gtts import gTTS
import openai
import os
import streamlit as st
from streamlit.components.v1 import html

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_gpt3_response(prompt):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are an assistant that specializes in Japanese logic and is an expert in presenting in Japanese. You provide hints, feedback, and logical corrections on user's questions, especially if they sound like they're from children. You have a veteran's experience in Japanese presentations and can teach in a gentle manner, step by step, tailored for elementary and junior high school students. Do not provide direct answers. Instead, guide them towards finding the answer themselves by suggesting ways or methods to research. If there are any inaccuracies or logical inconsistencies in the question, point them out and then guide the user on how and what general types of tools or resources they can use to find the correct information, without specifying the exact answer.Please ensure that the output always includes a list of logical expression errors in the question, along with the reasons why they are incorrect."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

# CSS for the chat bubbles
chat_style = """
<style>
    .chat-bubble {
        padding: 10px;
        border-radius: 15px;
        margin: 5px 0;
    }
    .user {
        background-color: #e1ffc7;
        align-self: flex-start;
    }
    .chatgpt {
        background-color: #c7c7ff;
        align-self: flex-end;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        width: 50%;
        margin: auto;
    }
</style>
"""

st.markdown(chat_style, unsafe_allow_html=True)

st.title('Teach Me')

# 会話の履歴を保存するリスト
conversation_history = []

user_input = st.text_input("作ったプレゼンの原稿や文章を入力してください:")

if user_input:
    response = get_gpt3_response(user_input)
    
    # ユーザーの質問とChatGPTの応答を履歴に追加
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "ChatGPT", "content": response})

    tts = gTTS(text=response, lang='ja')
    tts.save("response.mp3")

    # 会話の履歴を表示
    with st.markdown('<div class="chat-container">', unsafe_allow_html=True):
        for item in conversation_history:
            if item["role"] == "user":
                st.markdown(f'<div class="chat-bubble user">{item["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble chatgpt">{item["content"]}</div>', unsafe_allow_html=True)
    
    st.audio("response.mp3")
