import streamlit as st


def use_custom_css():
    with open("custom.css") as custom_css:
        return st.write(f'<style>{custom_css.read()}</style>', unsafe_allow_html=True)


def check_openai_key():
    if st.session_state.get("api_success", False) is False:
        st.warning("""
        No OpenAI key was found! If you don't set the OpenAI Key, none of the exercises here will work.
        """, icon="🤦‍♀️")
        with st.form("openai_key_form"):
            st.subheader("Enter your OpenAI API Key")
            st.text_input("OpenAI API Key", placeholder="sk-...", key="openai_key")

            submitted = st.form_submit_button("Submit")

            if submitted:
                from openai.error import AuthenticationError
                try:
                    import openai
                    openai.api_key = st.session_state.openai_key
                    openai.Model.list()
                except AuthenticationError:
                    st.session_state["api_success"] = False
                    st.error(
                        "An incorrect API Key was provided. You can find your API key at "
                        "https://platform.openai.com/account/api-keys."
                    )
                    return
                st.session_state["api_success"] = True
                st.success("Success! You are good to go.", icon="🎉")


def write_footer():
    st.divider()
    st.write(
        """
Prompt Engineering for Lawyers © 2023 by Ang Hou Fu is licensed under Attribution-ShareAlike 4.0 International  
[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/houfu/prompt-engineering-lawyers) 
        """
    )
