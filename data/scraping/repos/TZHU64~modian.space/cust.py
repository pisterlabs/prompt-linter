import streamlit as st
import openai
from streamlit_chatbox import *
import toml

openai.api_type = st.secrets["openai_api_type"]
openai.api_base = st.secrets["openai_api_base"]
openai.api_key = st.secrets["openai_api_key"]
openai.api_version = st.secrets["openai_api_version"]

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'message_log' not in st.session_state:
    st.session_state['message_log'] = []


def generate_response(message_log, temp, gpt_type):
    if gpt_type == "GPT-4":
        id = "gpt4"
    else:
        id = "gpt35"
    response = openai.ChatCompletion.create(
        deployment_id=id,
        messages=message_log,
        temperature=temp,
    )
    for choice in response.choices:
        if "text" in choice:
            return choice.text
    return response.choices[0].message.content


def cust_page():
    chat_box = ChatBox()
    chat_box.init_session()
    chat_box.output_messages()
    user_input = st.chat_input()

    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button("清空对话", use_container_width=True, ):
            chat_box.init_session(clear=True)
            st.experimental_rerun()

        export_btn.download_button(
            "导出记录",
            "".join(chat_box.export2md()),
            file_name=f"Chat.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown("""___""")
        st.markdown("定制聊天机器人的角色，角色描述的越详细，效果则会越好。下拉菜单中包含了一些预设的角色。")
        prompts = toml.load("./prompts.toml")
        selected = st.selectbox("select", prompts["prompts"], label_visibility="collapsed", )
        system_input = st.text_area("system_input", key="system_input", label_visibility="collapsed",
                                    value=prompts["prompts"][selected])
        system_button = st.button(":sunrise_over_mountains: 定制", key="system_button")
        st.markdown(
            "定制你喜欢的回答温度：0为最精确(即每次答案几乎一样)，1为最富有创造力(每次答案都区别较大), 默认为 0.7。")
        temp = st.number_input("回答温度：", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        gpt_type = st.selectbox("请选择模型:", ["ChatGPT (GPT-3.5)", "GPT-4"], index=0)

    if system_button:
        with st.spinner("定制中..."):
            chat_box.user_say(system_input)
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['message_log'].clear()
            st.session_state['message_log'] = [{"role": "system", "content": system_input}]
            output = generate_response(st.session_state['message_log'], temp, gpt_type)
            st.session_state['message_log'].append({"role": "assistant", "content": output})
            chat_box.ai_say(output)

    if user_input:
        with st.spinner("思考中..."):
            chat_box.user_say(user_input)
            st.session_state['message_log'].append({"role": "user", "content": user_input})
            output = generate_response(st.session_state['message_log'], temp, gpt_type)
            st.session_state['message_log'].append({"role": "assistant", "content": output})
            chat_box.ai_say(output)