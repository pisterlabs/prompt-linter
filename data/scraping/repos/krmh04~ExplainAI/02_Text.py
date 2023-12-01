import streamlit as st
import openai
import os
from text_summarizer.functions import summarize
st.session_state.update(st.session_state)
def main():
    st.title("Summarize your text in an instant!")
    input_text = st.text_area(label="Enter the text:", value="", height=250)
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
        if "summary" not in st.session_state:
            st.session_state["summary"] = ""
        
        if st.button("Submit", on_click=summarize, kwargs={"prompt": input_text}):
            st.write(st.session_state["summary"])
        
        
    except:
        st.write('There was an error =(')


if __name__ == '__main__':
    # LOGGED_IN key is defined by streamlit_login_auth_ui in the session state.
    if 'LOGGED_IN' in st.session_state and st.session_state.LOGGED_IN:
        main()
    else:
        st.write("Please login first")
