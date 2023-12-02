# This code does summarization

from langchain.document_loaders import UnstructuredFileLoader  # Importing necessary modules
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage
from langchain.prompts import PromptTemplate  # Importing PromptTemplate for prompts
import streamlit as st
import os
from html2docx import html2docx
import markdown
import openai
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pdfkit


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
         sadrzaj = infile.read()
         infile.close()
         return sadrzaj

def main():
         
    # Read OpenAI API key from env
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    # initial prompt
    prompt_string = open_file("prompt_summarizer.txt")
    prompt_string_pam = open_file("prompt_pam.txt")
    opis="opis"
    st.header('Zapisnik 16k')  # Setting the title for Streamlit application
    st.caption('Program radi sa turbo modelom sa 16k kontekstom.')
    st.caption('Date su standardne instrukcije koji mozete promeniti po potrebi.')
    st.caption("* dokumenti do velicine 6.000 karaktera ce biti tretirani kao jedan. Dozvoljeni formati su .txt. .docx i .pdf")

    uploaded_file = st.file_uploader("Choose a file")
    

    if 'dld' not in st.session_state:
        st.session_state.dld = "Zapisnik"

    # markdown to html
    html = markdown.markdown(st.session_state.dld)
    # html to docx
    buf = html2docx(html, title="Zapisnik")
    # create pdf
    options = {
    'encoding': 'UTF-8',  # Set the encoding to UTF-8
    'no-outline': None,
    'quiet': ''
    }
   
    pdf_data = pdfkit.from_string(html, False, options=options)
   
    # summarize chosen file
    if uploaded_file is not None:
        temp = st.slider('Set temperature (0=strict, 1=creative)', 0.0, 1.0, step=0.1)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=temp, openai_api_key=openai.api_key)  # Initializing ChatOpenAI model
    
        
        with st.form(key='my_form', clear_on_submit=False): 
               # st.write(uploaded_file.name)
               with open(uploaded_file.name, "wb") as file:
                        file.write(uploaded_file.getbuffer())
               if ".pdf" in uploaded_file.name:
                    loader = UnstructuredPDFLoader(uploaded_file.name, encoding="utf-8")
               else:
                    loader = UnstructuredFileLoader(uploaded_file.name, encoding="utf-8")  # Creating a file loader object

               result = loader.load()  # Loading text from the file

               text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=0)  # Creating a text splitter object
               texts = text_splitter.split_documents(result)  # Splitting the loaded text into smaller chunks
               
               opis = st.text_area("Unestite instrukcije: ", """Write a one page summary. Be sure to describe every topic and the name used in the text. Try to refer to speaker names. Write it as a bulletpoints and not as you are summarizing a text. Write only in Serbian language. Give it a Title and subtitles where appropriate and use caps lock to make what you marked as a title stand out.""", height=150)
               PROMPT = PromptTemplate(template=prompt_string, input_variables=["text", "opis"])  # Creating a prompt template object
               PROMPT_pam = PromptTemplate(template=prompt_string_pam, input_variables=["text"])  # Creating a prompt template object
               submit_button = st.form_submit_button(label='Submit')
               
              
               if submit_button:
                    with st.spinner("Sacekajte trenutak..."):
                        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True, map_prompt=PROMPT, combine_prompt=PROMPT_pam)
                            # Load the summarization chain with verbose mode
                        suma = AIMessage(content=chain.run(input_documents = texts, opis = opis))
                        st.session_state.dld= suma.content
                        html = markdown.markdown(st.session_state.dld)
                        buf = html2docx(html, title="Zapisnik")
                        pdf_data = pdfkit.from_string(html, False, options=options)   
        
        col1, col2, col3 = st.columns(3)                    
        with col1:
            st.download_button("Download as .txt", st.session_state.dld, file_name="zapisnik.txt")
        with col2:
          
            
            st.download_button(label="Download as .pdf",
                    data=pdf_data,
                    file_name="Zapisnik.pdf",
                    mime='application/octet-stream')
        with col3: 
            st.download_button(
                label="Download as .docx",
                data=buf.getvalue(),
                file_name="Zapisnik.docx",
                mime="docx"
            )
         
        with st.expander('Summary', True):
                           
                # Generate the summary by running the chain on the input documents and store it in an AIMessage object
                st.write(st.session_state.dld)  # Displaying the summary
                # update the content for all download formats
                          

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login to text summarizer', 'main')

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'main', key='unique_key')
    # if login success run the program
    main()
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')



