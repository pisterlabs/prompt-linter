import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index import download_loader
from llama_index.llms import OpenAI
import openai
import os
import ast
import re


def dict_from_string(response):
    """function to parse GPT response with competitors tickers and convert it to a dict"""
    # Find a substring that starts with '{' and ends with '}', across multiple lines
    match = re.search(r'\{.*?\}', response, re.DOTALL)

    dictionary = None
    if match:
        try:
            # Try to convert substring to dict
            dictionary = ast.literal_eval(match.group())
        except (ValueError, SyntaxError):
            # Not a dictionary
            return None
    return dictionary


# @st.cache_resource(show_spinner=False)
def load_data(file):
    PDFReader = download_loader("PDFReader", custom_path=os.getcwd())
    loader = PDFReader()

    docs = loader.load_data(file)
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0,
                                                              system_prompt="You are a legal expert and your job is to answer questions about NDA agreements in plain English."))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return index


st.set_page_config(page_title="ProSign - AI Powered NDA Review & Signing", page_icon="📝", layout="wide", menu_items=None)

page_bg_img = f"""
<style>
  /* Existing CSS for background image */
  [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/CxqMfWz4/bckg.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
  }}
  [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
  }}

  /* New CSS to make specific divs transparent */
  .stChatFloatingInputContainer, .css-90vs21, .e1d2x3se2, .block-container, .css-1y4p8pa, .ea3mdgi4 {{
    background-color: transparent !important;
  }}
</style>
"""

sidebar_bg = f"""
<style>
[data-testid="stSidebar"]{{
    z-index: 1;
}}
</style>
"""


st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(sidebar_bg, unsafe_allow_html=True)

openai.api_key = st.secrets["openai_credentials"]["openai_key"]
path = os.path.dirname(__file__)

# initiate session state values
if 'uploaded_file' not in st.session_state.keys():
    st.session_state['uploaded_file'] = None
if 'file_name' not in st.session_state.keys():
    st.session_state['file_name'] = None
if 'response_dict' not in st.session_state.keys():
    st.session_state['response_dict'] = None
if 'index' not in st.session_state.keys():
    st.session_state['index'] = None

st.title("ProSign - AI Powered NDA Review & Signing 📝")
st.write('')

if st.session_state['file_name'] is None:
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a document to get started 👇", type=["pdf"])

        # use sample file
        sample_toggle = st.toggle('Use sample file')
        
        st.link_button(label='Download Sample PDF', url='https://drive.google.com/file/d/11cXdPufz1nWc4qnsDdGlyQxtzDUkvM5N/view?usp=sharing')
        
        if sample_toggle:
            # load sample NDA
            uploaded_file = path + '//' + 'NDA_sample.pdf'

else:
    uploaded_file = None

if uploaded_file is not None and st.session_state['response_dict'] is None:
    with st.spinner(text="Loading and indexing the docs – hang tight!"):
        # for cases when sample file is used
        try:
            file_name = uploaded_file.name
        except AttributeError:
            file_name = "NDA_sample.pdf"

        index = load_data(file=uploaded_file)

        st.session_state['file_name'] = file_name
        st.session_state['index'] = index
        st.session_state['uploaded_file'] = uploaded_file

        chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

        prompt = """
        Answer the following questions about the document:

        Who are the signing parties?
        What is the effective date of the agreement?
        What is the duration of the agreement?
        What specifically constitutes "confidential information" under this agreement?
        What exactly are my obligations as the recipient of the confidential information?
        Are there any actions or activities restricted by this NDA?
        What are the consequences if there is a breach of the NDA?
        Does the document include any uncommon practices?

        Return output as Python dictionary with questions as keys and responses as values. 
        Do not use comma as thousands separator.
        Be precise and concise. Use simple English.
        """

        # get response from GPT
        response = chat_engine.chat(prompt)

        # convert response to dict
        response_dict = dict_from_string(response.response)
        st.session_state['response_dict'] = response_dict

    st.header('Summary of {}'.format(file_name))
    st.info('Got specific questions? Go to the Chat with Doc tab on the left!')

    for k, v in response_dict.items():
        st.subheader(k)
        st.write(v)
        st.divider()

if uploaded_file is None and st.session_state['response_dict'] is not None:
    st.header('Summary of {}'.format(st.session_state['file_name']))
    st.info('Got specific questions? Go to the Chat with Doc tab on the left!')

    for k, v in st.session_state['response_dict'].items():
        st.subheader(k)
        st.write(v)
        st.divider()

footer_html = """
    <div class="footer">
    <style>
        .footer {
            position: fixed;
            z-index: 2;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #283750;
            padding: 10px 20px;
            text-align: center;
        }
        .footer a {
            color: #4a4a4a;
            text-decoration: none;
        }
    </style>
        Made for Dropbox Sign AI Hackathon 2023. Powered by LlamaIndex and OpenAI.
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
