import streamlit as st
import PyPDF2
import openai
from io import BytesIO
import os
from text_summarizer.functions import summarize
st.session_state.update(st.session_state)
def main():
    st.title('Summarize your PDF!')
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if "summary" not in st.session_state:
            st.session_state["summary"] = ""
        uploaded_file = st.file_uploader('Choose a PDF file', type=['pdf'])
        if uploaded_file is not None:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
        # text = extract_text(uploaded_file)
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text().lower()
            if st.button("Submit", on_click=summarize, kwargs={"prompt": page_text}):
                page_text =st.write(st.session_state["summary"])

            
    except Exception as e:
        st.write('There was an error =(')
if __name__ == '__main__':
    # LOGGED_IN key is defined by streamlit_login_auth_ui in the session state.
    if 'LOGGED_IN' in st.session_state and st.session_state.LOGGED_IN:
        main()
    else:
        st.write("Please login first")