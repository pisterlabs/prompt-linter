import streamlit as st
import base64
from io import BytesIO
import openai
import os
from text_summarizer.functions import summarize
from streamlit.components.v1 import html
st.session_state.update(st.session_state)

def main():
    # display the upload form
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    st.markdown("Please **right click** or use **CTRL + C** to copy the text")
    st.markdown("Please **right click** or use **CTRL + V** to paste the text in the textbox")
    
    # if a file is uploaded
    if uploaded_file is not None:
        # load the PDF file using requests and display it in the app
        pdf_data = uploaded_file.read()
        b64 = base64.b64encode(pdf_data).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
        input_text = st.text_area(label="Enter the text:", value="", height=250)
        try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
            
                if "summary" not in st.session_state:
                    st.session_state["summary"] = ""
                
                if  st.button("Submit", on_click=summarize, kwargs={"prompt": input_text}):
                    st.write(st.session_state["summary"])
                
                
        except:
                st.write('There was an error =(')
# Wrapt the javascript as html code
if __name__ == '__main__':
    # LOGGED_IN key is defined by streamlit_login_auth_ui in the session state.
    if 'LOGGED_IN' in st.session_state and st.session_state.LOGGED_IN:
        main()
    else:
        st.write("Please login first")        
