import redis
import os
import streamlit as st
import openai
import numpy as np
from redis.commands.search.query import Query
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
from rich import print

load_dotenv()
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_TEXT_MODEL = "gpt-3.5-turbo"
INDEX_NAME = "idx:blogs"
CHAT_HISTORY = "streamlit:chat:history"
openai.api_key = os.getenv("OPENAI_API_KEY")
EXPLANATION = []
# Common Functions

def get_explanation():
    expl_doc = ""
    for i, txt in enumerate(EXPLANATION):
        expl_doc += f"{i+1} : {txt}<br><br>"
    return expl_doc

def sticky():
    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid #A0A2E4;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )

def fixed_bottom():
    # make footer fixed.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )

def center_columns():
    st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(2)
                {
                    text-align: center;
                } 
            </style>
            """,
            unsafe_allow_html=True,
        )

def get_redis_conn() -> redis.Redis:
    redis_host, redis_port, redis_user, redis_pass = (
        os.getenv("redis_host"),
        os.getenv("redis_port"),
        os.getenv("redis_user"),
        os.getenv("redis_pass"),
    )
    if not redis_user:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    else:
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            username=redis_user,
            password=redis_pass,
            decode_responses=True,
        )
    return r


def render_chat_history(r: redis.Redis, stream):
    if r.exists(stream):
        chat_history_msgs = r.xrange(stream)
        for ts, message in chat_history_msgs:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)


def clear_chat_history(r: redis.Redis):
    print("Clearing Chat History")
    keys = [key for key in r.scan_iter(match=f"{CHAT_HISTORY}*")]
    if keys:
        r.delete(*keys)


def main():
    r = get_redis_conn()
    st.set_page_config()

    with st.container():
        colored_header(
            label="LLM Chatbot without RAG",
            description="'Out of date' information and 'hallucinations'",
            color_name="violet-60",
        )
        # st.header(body="LLM Chatbot without RAG", divider="violet")
        col1, col2, col3 = st.columns(3)
        center_columns()
        col2.button(
            "Clear Chat History",
            type="primary",
            on_click=clear_chat_history,
            kwargs={"r": r},
        )
        expl = st.expander(label="Execution Log")
        sticky()

    with st.container():
        render_chat_history(r, stream=CHAT_HISTORY)

        if prompt := st.chat_input("Ask me anything!"):
            with st.chat_message("user"):
                st.markdown(prompt)
                r.xadd(CHAT_HISTORY, {"role": "user", "content": prompt})
                EXPLANATION.append(f"Prompt entered by the user : '{prompt}' ")

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in openai.ChatCompletion.create(
                    model=OPENAI_TEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                EXPLANATION.append(f"Open AI *{OPENAI_TEXT_MODEL}* responds with a generated response")

                r.xadd(CHAT_HISTORY, {"role": "assistant", "content": full_response})

            expl.markdown(get_explanation(), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
