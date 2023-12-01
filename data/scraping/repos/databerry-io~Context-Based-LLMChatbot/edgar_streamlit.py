#earning report example: https://www.sec.gov/Archives/edgar/data/1288776/000119312511011442/dex991.htm
#earning report api: https://site.financialmodelingprep.com/developer/docs/earning-call-transcript-api/
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import chat
import streamlit as st
from streamlit_chat import message
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Iterator
from langchain.schema import Document

from pypdf import PdfReader

import config

import os
import json
from pathlib import Path
import pdfkit
import fpdf
from fpdf import FPDF
from bs4 import BeautifulSoup
import html
import fmp_endpoint
#import weasyprint

st.set_page_config(page_title="DOCCHAT | WITHMOBIUS", 
                #    initial_sidebar_state="collapsed",
                   page_icon="data:image/png;base64,/9j/4AAQSkZJRgABAQIAHAAcAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAAgACADAREAAhEBAxEB/8QAGQAAAgMBAAAAAAAAAAAAAAAABAcFBggJ/8QAMBAAAgEDAwEECAcAAAAAAAAAAQIDBAURAAYSIQcIIjETFBUyQVFhcSNDYoGRkqH/xAAZAQACAwEAAAAAAAAAAAAAAAACBAEFBgD/xAAlEQACAgECBgMBAQAAAAAAAAABAgADBBEhBRITIlGBMUGRMtH/2gAMAwEAAhEDEQA/AOVWunSfoOz7fd0jEtu2be6mMjkGioJWBH0PHro1qsfdVJ9Q1rdt1Un1Arttjclhx7dsFxt3I4HrdLJDn+wGoZWU6MNIJUrsRpI3QyIzuyW10UdtuN+eSCK6tURUlqabiFJCs8/Fm6JJxMQVjj3mAIJGnsBVNnM41AjuCqmzmYagRu2251FinjO6KyntksgDAXGsjgdgRkHEjAnpjrrTJn0V9rOBNGudQnazgR4bH3HS3OgVDUUlyt8+Y2AljqqeT5qQCyHz6g/TTy2UZabEMP2MB6cpdiGH7FD3r+7ps207Oftd7OqGK0CjmiivFsh6U5SVgiTwr+WQ5VWQeEhgVC4IOa4tw1McdarYfY/yZ7iWCtA6tew8TPVveSTatr9ko88dG9S9esQ5PDK7rhmUdeJjSMBvLIYZyMaqsezptEMezkaW7a+6qmmUU61xRBkGIuCgwOvhPTzPy+GtFQyWDuGsv6WSwdw1jm2fueWOn9KzJDSRfiPJxSKFMgZdm8KDyGST8BqzranGXm2UehH1NNC67KPQlJ7wveJtm49oP2ZbRqRWwVU0cl0rlz6IrE3JIYiffHMKzPgDwqFyMnWe4txNMoCmn+fs+ZQ8Tz0yB0qvjz5mbqaqqaOZaikqJIJU9143KsPsR1GqOU8mYd+bwgXim4aw/qZ+Tfycn/dGLHX4Jhix1+CYBcr/AHy88RdrxW1gTqonnaQL9gTgftoWYtux1gli25MA1Eif/9k=", layout="wide")

#Creating the chatbot interface
#st.title("Mobius: LLM-Powered Chatbot")
st.markdown("""
<h1 style='text-align: center; color: teal;'>Mobius: LLM-Powered EDGAR Chatbot</h1>
<style>
    .katex .base {
        width: 100%;
        display: flex;
        flex-wrap: wrap;
    }
    .stCodeBlock code {
        white-space: break-spaces !important;
        }
</style>
""", unsafe_allow_html=True)



# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'citation' not in st.session_state:
    st.session_state['citation'] = []

# Define a function to clear the input text
def clear_input_text():
    global input_text
    input_text = ""

# We will get the user's input by calling the get_text function
def get_text():
    global input_text
    input_text = st.text_input("Select Company Ticker", key="input1", on_change=clear_input_text)
    return input_text


# Define a function to clear the input text
def clear_input_question():
    global input_question
    input_question = ""

# We will get the user's input by calling the get_text function
def get_question():
    global input_question
    input_question = st.text_input("Input Your Question", key="input2", on_change=clear_input_question)
    return input_question

# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

def company_info():
    with open("/Users/xiang/PycharmProjects/Context-Based-LLMChatbot/kaggle/company_tickers_exchange.json", "r") as f:
        CIK_dict = json.load(f)

    # convert CIK_dict to pandas
    CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])

    return CIK_df

def edgar_api():
    result  = ''
    CIK_df = company_info()
    CIK = CIK_df[CIK_df["ticker"] == user_select].cik.values[0]

            # preparation of input data, using ticker and CIK set earlier
    url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"

    # read response from REST API with `requests` library and format it as python dict
    import requests
    header = {
    "User-Agent": ""#, # remaining fields are optional
    #    "Accept-Encoding": "gzip, deflate",
    #    "Host": "data.sec.gov"
    }

    company_filings = requests.get(url, headers=header).json()
    company_filings_df = pd.DataFrame(company_filings["filings"]["recent"])
    access_number = company_filings_df[company_filings_df.form == "10-K"].accessionNumber.values[0].replace("-", "")
    file_names = company_filings_df[company_filings_df.form == "10-K"].primaryDocument.values

    for file_name in file_names:

        url_file = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{file_name}"
        print(f"url_file is {url_file}")

        # dowloading and saving requested document to working directory
        req_content = requests.get(url_file, headers=header).content.decode("utf-8")

        soup = BeautifulSoup(req_content, 'html.parser')
        req_content_text = soup.get_text()
        result += req_content_text

        return result



EAR_API_KEY = ""

def main():

    CIK_df = company_info()
    col_one_list = CIK_df["ticker"].tolist()

    unique_sorted = sorted(list(set(col_one_list)))
    forms_list = ['10-K', 'earnings conference call']
    years_list = [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013]
    quarters_list = ['FY',1,2,3,4]

    with st.sidebar:
        user_select = st.selectbox('Select Company Ticker', unique_sorted)
        user_select_forms = st.selectbox('Select File Form', forms_list)
        user_select_year = st.selectbox('Select Year', years_list)
        user_select_quarter = st.selectbox('Select Quarter', quarters_list)
        user_question = get_question()

    st.markdown("""---""")

    if user_select and user_select_forms and user_select_year:
        year = user_select_year

        req_content_text_all = ""
        if user_select_forms == "10-K":
            quarter = "FY"
            url = f"https://financialmodelingprep.com/api/v4/financial-reports-json?symbol={user_select}&year={year}&period={quarter}&apikey={EAR_API_KEY}"
            result = fmp_endpoint.get_jsonparsed_data(url)
            req_content_text = str(result)
            req_content_text_all += req_content_text

        else:
            if user_select_quarter and user_select_quarter != "FY":
                quarter = user_select_quarter
            else:
                quarter = 1
            
            if year >= 2021:
                print("using edgar api")
                result = edgar_api()
                req_content_text_all += result
            else:
                print("using fmp api")
                url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{user_select}?quarter={quarter}&year={year}&apikey={EAR_API_KEY}"
                result = fmp_endpoint.get_jsonparsed_data(url)
                req_content_text = result[0]["content"]
                req_content_text_all += req_content_text
            


        if user_question:
            pages = text_to_docs(req_content_text_all)
            output, sources = chat.answer_Faiss_rate(user_question, pages)

            st.session_state.past.append(user_question)
            st.session_state.generated.append(output)
            converted_sources = [doc.page_content for doc in sources]
            st.session_state.citation.append(converted_sources)

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.title("Chat")
                with col2:
                    st.title("Citation")


    with st.container():
        col1, col2 = st.columns(2, gap="large")
        #print("session is: ", st.session_state)
        required_keys = ['generated', 'past', 'citation']

        if all(st.session_state.get(key) for key in required_keys):

            for i in range(len(st.session_state['generated'])-1, -1, -1):
                with col1:
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))
            with col2:
                #for item in st.session_state["citation"][-1]:
                for i in range(len(st.session_state['citation'])-1, -1, -1):
                    st.info(st.session_state['citation'][i], icon="ℹ️")
                    #st.info(str(item), icon="ℹ️")




# Run the app
if __name__ == "__main__":
    main()