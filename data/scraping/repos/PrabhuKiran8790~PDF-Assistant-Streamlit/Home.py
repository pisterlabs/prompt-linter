import streamlit as st
from components.sidebar.OpenAI_API import openai_api_insert_component
from components.body.file_uploader import file_uploader
from components.body.prompt import prompt_box
from components.body import langchain_PDF
from components.sidebar.Auth import authentication_comp, db
import pandas as pd
import os


st.set_page_config(page_title="PDF Assistant", page_icon="📖", layout="wide", initial_sidebar_state='expanded')

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'username' not in st.session_state:
    st.session_state['username'] = None

if 'login_btn_clicked' not in st.session_state:
    st.session_state['login_btn_clicked'] = None
    
if 'uuid' not in st.session_state:
    st.session_state['uuid'] = None
    
if 'login_failed' not in st.session_state:
    st.session_state['login_failed'] = None
    
if 'response' not in st.session_state:
    st.session_state['response'] = None


def main():
    st.header(":red[PDF Assistant]: AI-Powered Q&A for _PDFs_")
    
    if st.session_state['logged_in'] != False and st.session_state['username'] is not None:
        st.sidebar.write(f"Welcome **:green[{st.session_state['username']}]** 👋")
    
    # st.write(os.getenv("FIREBASE_API"))
    openai_api_insert_component() # Insert OpenAI API component in sidebar
    
    # if not logged in, show authentication component
    if st.session_state['logged_in'] == False:
        with st.sidebar:
            authentication_comp()
    
            
    # if logged in, show logout button
    if st.session_state['logged_in'] == True:
        with st.sidebar:
            logout = st.button("Logout 🔒")
            if logout:
                st.session_state['logged_in'] = False
                st.session_state['login_btn_clicked'] = None
                st.session_state['username'] = None
                st.session_state['uuid'] = None
                st.session_state['signup_btn_clicked'] = None
                st.button("dummy", on_click=st.experimental_rerun()) # dummy button to rerun the app. This is a hacky way to rerun the app. dummy btn is not shown to user.
                
                
    file_uploader_col, prompt_col = st.columns([0.5, 1])
    with file_uploader_col:
        file_uploader()
    with prompt_col:
        prompt_box()
    
    
    generate_answer_button = st.button("Generate Answer")
    if generate_answer_button:        
        
        st.session_state['generate_answer_button'] = True
        
        # check if all are empty
        if st.session_state['OPENAI_API_KEY'] == "" and st.session_state['uploaded_file'] is None and st.session_state['prompt'] == "":
            st.error("Please set your OpenAI API key in the sidebar, upload a PDF and enter a prompt")
            st.session_state['cancel_btn_active'] = True
            # st.stop()
        
        # check if API key is empty 
        elif st.session_state['OPENAI_API_KEY'] == "" or st.session_state['OPENAI_API_KEY'] is None:
            st.sidebar.error("Please set your OpenAI API key in the sidebar.")
            st.session_state['cancel_btn_active'] = True
            # st.stop()
        
        # check if file is not uploaded and prompt is empty
        elif st.session_state['uploaded_file'] is None and st.session_state['prompt'] == "":
            st.error("Please upload a PDF and enter a prompt")
            st.session_state['cancel_btn_active'] = True
            # st.stop()

        # check if file is not uploaded
        elif st.session_state['uploaded_file'] is None:
            st.error("Please upload a PDF")
            st.session_state['cancel_btn_active'] = True
            # st.stop()
        
        # check if prompt is empty
        elif st.session_state['prompt'] == "":
            st.error("Please enter a prompt")
            st.session_state['cancel_btn_active'] = True
            # st.stop()
        
        else: # if everything is fine
            os.environ['OPENAI_API_KEY'] = st.session_state['OPENAI_API_KEY']
            st.caption(f"Filename: :red[{st.session_state['uploaded_file'].name}]")
            response = langchain_PDF.get_response_from_OpenAI_LangChain(st.session_state['uploaded_file'], st.session_state['prompt'])
            # st.session_state['response'] = response
            st.warning('⚠️ Please note that the response is dependent on the :red[Quality of the PDF] and the :red[Quality of the prompt] and it may not be accurate at times. Please use the response as a reference and not as a final answer.')
            
            
    if st.session_state['response'] is not None:
        st.write("")
        st.write("###### :blue[🤖 **AI Response**]")
        st.write(f"#### :green[{st.session_state['response']}]")
        st.markdown("------------")
    
    if st.session_state['logged_in'] == True and st.session_state['username'] is not None:
        show_history = st.checkbox("Show History")
        
        if show_history:
            st.write("Your previous interactions are as follows:")
            past_docs = db.child("users").child(st.session_state['uuid']).child('pdf_files').get().val()
            if past_docs:
                selected_doc = st.selectbox("Select a PDF file", options=list(past_docs.keys()))
                df = pd.DataFrame.from_dict(past_docs[selected_doc]['Prompts'], orient='index', columns=['prompt', 'response'])
                hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """
                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                st.table(df)

            else:
                st.write("##### 😔 :red[No history found.]")
                
if __name__ == "__main__":
    main()
    