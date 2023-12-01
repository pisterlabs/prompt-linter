import openai
import streamlit as st
import tiktoken
import os
from dotenv import load_dotenv

from projects.home.definitions import lab_contributors
from projects.home.utils import contributor_card

load_dotenv()


def main():

    st.set_page_config(
        page_title="ERNI Data & AI Community Lab",
        page_icon="💬",
        #layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": """This is a demo Streamlit app made by the ERNI's Data and AI Community.
                        Code: https://github.com/enricd/erni_data_ai_community_lab/"""
        }
    )


    # --- Side Bar ---
    with st.sidebar:
        openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        user_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=openai_api_key, type="password")
        if user_api_key != "":
            openai.api_key = user_api_key

        model = st.selectbox("Select a model:", ["gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], index=0)
        encoding = tiktoken.encoding_for_model(model)

        if "tokens_count" not in st.session_state:
            st.session_state["tokens_count"] = {"prompt": 0, "completion": 0, "total": 0}

        st.write("Session Tokens Count (aprox.):")
        tokens_count = st.empty()
        tokens_count.write(st.session_state.tokens_count)

        st.button("Reset Conversation", 
                  on_click=lambda: st.session_state.pop("messages", None) if "messages" in st.session_state and len(st.session_state.messages) > 0 else None,
                  type="primary",
                  )

        st.divider()

        st.markdown("## Project Contributors:")
        # Create a card for each contributor
        for contributor in ["enricd"]:  
            st.markdown(contributor_card(
                **lab_contributors[contributor],
                ), 
                unsafe_allow_html=True)


    # --- Main Page ---
    st.markdown("<h1 style='text-align: center;'>🤖💬 <em>LLM Chatbot</em></h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hi! Ask me anything?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Requesting completion to the OpenAI API
            print("model used:", model)
            for response in openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Update tokens counter
        st.session_state.tokens_count["prompt"] += sum([len(encoding.encode(m["role"] + m["content"])) + 4 for m in st.session_state.messages])
        st.session_state.tokens_count["completion"] += len(encoding.encode(full_response))
        st.session_state.tokens_count["total"] = st.session_state.tokens_count["prompt"] + st.session_state.tokens_count["completion"]
        tokens_count.write(st.session_state.tokens_count)


if __name__ == "__main__":
    
    main()