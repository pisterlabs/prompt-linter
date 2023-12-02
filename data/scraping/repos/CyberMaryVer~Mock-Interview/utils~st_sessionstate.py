import os
import streamlit as st
from ai_analysis.openai_tools import OPENAI_KEY, check_if_key_valid


def _enter_key():
    """
    This function gets user key
    """
    with st.form("key_form"):
        st.markdown("🔑 **Enter your key**")
        example_input = "Enter key"
        user_input = st.text_input("st-xxxx",
                                   max_chars=500,
                                   placeholder=example_input,
                                   label_visibility="collapsed")
        submitted = st.form_submit_button("Submit")
        if submitted:
            is_valid = check_if_key_valid(user_input)
            st.markdown(f"☑️ :green[Key is valid]") if is_valid \
                else st.markdown(f"❌ :red[Key is invalid]")

            if is_valid:
                st.session_state["api_key"] = user_input
                os.environ["OPENAI_API_KEY"] = user_input


def st_getenv(key, default=None):
    """
    This function gets environment variable
    """
    if key in st.session_state:
        return st.session_state[key]
    else:
        return default


def st_apikey():
    # Check environment variable
    placeholder = st.empty()
    if OPENAI_KEY is not None:
        st.session_state['api_key'] = OPENAI_KEY
        os.environ["OPENAI_KEY"] = OPENAI_KEY
    if st_getenv('api_key') is None:
        st.warning("Please add your OpenAI API key to continue.")
        with placeholder:
            _enter_key()
        print("API key set")
    if st_getenv('api_key') is not None:
        print(f"\033[093mAPI key: {st.session_state['api_key'][:6]}...\033[0m")
        with placeholder:
            st.markdown("☑️ :green[API key set]")
