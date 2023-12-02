import streamlit as st
import openai
import os
from text_summarizer.functions import summarize
import scraper as scr
st.session_state.update(st.session_state)
def main():
    st.markdown(""" 
    <style>
    .big-font {
    font-size:37px !important;
    font-weight:bold !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Summarize your text from any weblink</p>',unsafe_allow_html=True)
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')

        if "summary" not in st.session_state:
            st.session_state["summary"] = ""

        input_url = st.text_input("Enter the URL:")

        if st.button("Submit", on_click=summarize, kwargs={"prompt": input_url}):

            scraper = scr.Scraper()
            response = scraper.request_url(input_url) 
            input_url = (
                scraper.extract_content(response)[:6000].strip().replace("\n", " ")
            )
            st.write(st.session_state["summary"])

    except Exception as e:
        st.write('There was an error =(')
if __name__ == '__main__':
    # LOGGED_IN key is defined by streamlit_login_auth_ui in the session state.
    if 'LOGGED_IN' in st.session_state and st.session_state.LOGGED_IN:
        main()
    else:
        st.write("Please login first")