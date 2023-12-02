import time

import streamlit as st
from friend_replica.format_chat import ChatConfig, format_chat_history, split_chat_data
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *
from langchain.llms import GPT4All

# from models.model_cn import ChatGLM



### Side Bar Module ###
with st.sidebar:
    "[Get a Comma API key](https://github.com/roxie-zhang/friend_replica)"
    "[View the source code](https://github.com/roxie-zhang/friend_replica)"

### Header Module ###
st.title("Comma Semantic Search")
st.caption("🚀 Semantic search is a powerful information retrieval technique that "
           "aims to enhance the accuracy and relevance of search results by understanding the character and chat contect of your chat history. "
           "| *FDU Comma Team Ver-1.1*")
# st.markdown('---')

### Config Model ###
st.subheader('Semantic Key word')
st.caption("Try keywords like happy, sad, angry and more! Don't worry, this is all private local LLM!")
config_form = st.form('Config Model')
col1, col2 = config_form.columns(2)
date_start =col1.date_input("Search Start Date", datetime.date(2023, 7, 1))
date_end = col2.date_input("Search End Date", datetime.date(2019, 10, 1))
queries = config_form.text_input('Prompts:')
button = config_form.form_submit_button('Start Search')

### Def ###
@st.cache_resource
def semantic_query(queries):
    contexts, msgs = st.session_state.chat_with_friend.semantic_search_with_org_msg(queries, k=10, threshold=.5, debug=False)
    return contexts, msgs


if "messages" not in st.session_state:
    st.session_state.messages = []
#
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


if "chat_blocks" not in st.session_state:
    st.warning("It seems you have not config the model yet. Please first config your replica agent in the main page")
else:
    if button:
        chat_container = st.container()
        with chat_container:
            try:
                contexts, msgs = semantic_query(queries)
            except Exception as e:
                st.warning('Query Body Error')
                st.warning(e)
            
            if isinstance(msgs, list):
                for i in range(len(msgs)):
                    if 'content' in msgs[i][0].keys():
                        continue
                    if len(msgs[i]) > 0:
                        msgs[i] = format_chat_history(msgs[i], st.session_state.chat_with_friend.chat_config, time=True)
                    else:
                        continue
            else:
                msgs = None
                contexts = None
            
            try:
                for i, context in enumerate(contexts):
                    with st.expander('\n\n'.join([msg['role'] + ': ' + msg['content'] for msg in msgs[i]])):
                        format_context = format_chat_history(context, 
                                                            st.session_state.chat_with_friend.chat_config,
                                                            time=True,
                                                            )
                        # st.write(format_context)
                        for each in format_context:
                            # st.session_state.messages.append({"role": role, "content": time+':'+content})
                            with st.chat_message(each['role']):
                                st.caption(each['time'])
                                st.markdown(each['content'])
            except Exception as e:
                st.warning('Not Found')

### End of the page ###
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown('---')
st.markdown('> *This demo version is made by Zihan for Learning use only*')