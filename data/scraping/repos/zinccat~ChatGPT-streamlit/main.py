import openai
import streamlit as st
from scripts.key import openai_key, anthropic_key, hf_key
from scripts.show import show_history, reset
from scripts.misc import generate_csv, init
from scripts.style import set_style
import numpy as np
import sys
import os
import csv
from PIL import Image
import time
from pathlib import Path
from scripts.models import *
from prompts import get_prompt, get_prompt_list

set_style()
init()

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
with st.sidebar:
    st.sidebar.title("ChatGPT")
    model_name = st.selectbox("选择模型:", get_model_list())
    prompt_name = st.selectbox("请选择提示词:", get_prompt_list())
    stream_mode = st.radio("是否使用流式输出:", ("Yes", "No"), horizontal=True)
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.button("清除对话历史", key="clear")
    # with st.form(key='key_form', clear_on_submit=True):
    #     key_input = st.sidebar.text_area("输入自己的apikey:", key='key_input', height=20)
    #     key_submit_button = st.sidebar.form_submit_button(label='Send')
    st.download_button('下载对话历史 (好像有bug, 多点两次)', generate_csv(), 'history.csv', key='download_button')
    temperature = st.slider(
        "模型温度",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=1e-1,
    )
    userkey = st.text_input("输入APIKEY (可选, 避免rate limit):", type="password")
    if userkey != '':
        openai.api_key = userkey
        anthropic_key = userkey
    st.markdown("<h3 style='text-align: center;'>Powered by <a href='https://www.github.com/ZincCat/'>ZincCat</a></h3>", unsafe_allow_html=True)

if clear_button:
    reset()

# set page title
if "SD" not in model_name:
    st.markdown(f"<h1 style='text-align: center;'>{prompt_name}</h1>", unsafe_allow_html=True)
else:
    st.markdown(f"<h1 style='text-align: center;'>{model_name}</h1>", unsafe_allow_html=True)

use_promptsafe = True
if use_promptsafe:
    sys.path.append("./PromptSafe")
    from promptsafe import promptfilter
    embedding_path = './embeddings'

model = get_model(model_name)

if st.session_state['generated']:
    show_history()

ques_box = st.empty()
res_box = st.empty()

# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("##### 🧙输入:", key='input', height=250, max_chars=36000)
        c_1, c_2, c_3, c_4 = st.columns(4)
        with c_1:
            submit_button = st.form_submit_button(label='发送', use_container_width=True)
        with c_2:
            clean_button = st.form_submit_button(label='清空', use_container_width=True)
        with c_3:
            regenerate_button = st.form_submit_button(label='重新生成', use_container_width=True)
        with c_4:
            reset_button = st.form_submit_button(label='清除对话历史', use_container_width=True)

    if regenerate_button:
        if st.session_state['generated']:
            submit_button = True
            user_input = st.session_state['past'][-1]
            st.session_state['generated'] = st.session_state['generated'][:-1]
            st.session_state['past'] = st.session_state['past'][:-1]
            st.session_state['model_name'] = st.session_state['model_name'][:-1]
            st.session_state['messages'] = st.session_state['messages'][:-2]
    if submit_button and user_input:
        with ques_box.container():
            st.info(f'''{user_input}''', icon="🧙")
        st.session_state['messages'][0] = {"role": "system", "content": get_prompt(prompt_name)}
        st.session_state['messages'].append({"role": "user", "content": user_input})
        output = ""
        if "SD" in model_name:
            if model_name == 'Openjourney (SD)':
                user_input = ' ,'.join((user_input, 'mdjrny-v4 style'))
            with res_box.container():
                with st.spinner('Wait a sec... (This may take a while if the model not loaded yet)'):
                    output = sd_generate_response(model)
                    while output == 503: # model not loaded yet
                        time.sleep(3)
                        output = sd_generate_response(model)
                    if isinstance(output, Image.Image):
                        st.image(output)
                        st.session_state['generated'].append(output)
                        st.session_state['past'].append(user_input)
                    elif output == 429:
                        st.error(f'''Too many requests! Please try again later.''', icon="🤖")
                    else:
                        st.error(f'''{output}''', icon="🤖")
        else:
            if 'GPT' in model_name:
                if stream_mode == 'No':
                    # output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
                    with res_box.container():
                        with st.spinner('Wait a sec...'):
                            output = gpt_generate_response(model, temperature)
                else:
                    report = []
                    # Looping over the response
                    response = gpt_generate_response_stream(model, temperature)
                    for chunk in response:
                        chunk_message = chunk['choices'][0]['delta']  # extract the message
                        if 'content' in chunk_message.keys():
                            report.append(chunk_message['content'])
                            output = "".join(report).strip()
                            with res_box.container():
                                st.success(f'''{output}''', icon="🤖")
            elif "Claude" in model_name:
                if stream_mode == 'No':
                    with res_box.container():
                        with st.spinner('Wait a sec...'):
                            output = claude_generate_response(model, temperature)
                else:
                    report = []
                    response = claude_generate_response_stream(model, temperature)
                    for chunk in response:
                        output = chunk['completion']    
                        with res_box.container():
                            st.success(f'''{output}''', icon="🤖")
            if use_promptsafe:
                prompt_embedding = np.load(os.path.join(embedding_path, '{}.npy'.format(prompt_name)))
                filter_flag, similarity = promptfilter(output, prompt_embedding, threshold=0.93)
                if filter_flag:
                    output = "You shall not pass!!!"
            with res_box.container():
                st.success(f'''{output}''', icon="🤖")
            st.session_state['messages'].append({"role": "assistant", "content": output})
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
    if reset_button:
        reset()

# with st.expander("Changelog"):
#     st.write("Todo: error handling and history summarization.")
#     st.write("2021-04-16: Added Stable Diffusion.")
#     st.write("2023-04-15: Added PromptSafe, Claude, history, and stream mode.")