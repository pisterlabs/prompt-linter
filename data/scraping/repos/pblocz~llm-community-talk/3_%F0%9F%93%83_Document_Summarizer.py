from pathlib import Path
from urllib.parse import unquote, urlparse
import requests
from bs4 import BeautifulSoup
import streamlit as st

import pdfplumber
import functools as ft
import openai

from modules.llm import summarize_text
from modules.state import read_url_param_values

import os
from Home import APP_TITLE, APP_ICON


st.set_page_config(
    page_title=f"{APP_TITLE} - Document summarize",
    page_icon=APP_ICON
)


def extract_page_lines(file, page_range):
    minp, maxp = page_range or [1, -2]
    with pdfplumber.open(file) as pdf:
        extracted_text = []
        for page in pdf.pages[minp - 1: maxp + 1]:
            page_text = page.extract_text(layout=True)
            extracted_text.append([i.strip() for i in page_text.splitlines()])
    return extracted_text


def is_pdf(file):
    return file is not None and file.type == "application/pdf"


def get_pages_lines(file, page_range):
    result_file = file
    if is_pdf(file):
        return extract_page_lines(file, page_range=page_range)

    return [result_file.read().splitlines()]


def get_pdf_pages(file):
    if is_pdf(file):
        with pdfplumber.open(file) as pdf:
            return len(pdf.pages)
    return -1


@st.cache_data()
def process_pdf(uploaded_file, page_range, clean_document):
    pages = get_pages_lines(uploaded_file, page_range=page_range)
    if clean_document:
        hfooter = ft.reduce(set.intersection, (set(page) for page in pages[:10]))
        pages = [[line for line in page if line not in hfooter] for page in pages]

    pages_text = ["\n".join(page) for page in pages]
    text = "\n\n".join(pages_text)
    return text


@st.cache_data()
def process_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text


@st.cache_data()
def summarize_document(text, number_of_chunks):
    return summarize_text(text, number_of_chunks, include_costs=True)


def configuration():
    # Config
    config = read_url_param_values()
    api_key = config["api_key"]
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


def get_text_and_summary_file(uploaded_file, url, page_range, clean_document):
    if url:
        summary_file = unquote(Path(urlparse(url).path).name)
        text = process_url(url)

    elif is_pdf(uploaded_file):
        summary_file = uploaded_file.name
        text = process_pdf(uploaded_file, page_range, clean_document)
    else:
        summary_file = uploaded_file.name
        text = uploaded_file.getvalue().decode()
    return summary_file, text


"""
# Document summarizer
"""
configuration()

"""
### Please provide either a file or a url
"""

# Pdf input
uploaded_file = st.file_uploader("Upload a document to summarize, 10k to 100k tokens works best!", type=['txt', 'pdf'])
max_pages = get_pdf_pages(uploaded_file)

# Url input
url = st.text_input("Url of the page to summarize")

# Validate Pdf and url input
if uploaded_file and url:
    st.error("Please, provide only file or url")
    st.stop()

# Pdf extra configuration
page_range = None
clean_document = False
if is_pdf(uploaded_file):
    with st.expander("For this file, how should it be processed"):
        if max_pages > 0:
            page_range = st.slider("Select page range to summarize", value=[1, max_pages], min_value=1, max_value=max_pages)

        if is_pdf(uploaded_file):
            clean_document = st.checkbox("Try to remove common footers and headers")

"""
### Summarization configuration
"""
number_of_paragraphs = st.number_input(label="Number of paragraphs in summary", min_value=1, max_value=10, value=1)


if st.button("Summarize file", type="primary"):
    with st.spinner("SummarAIzing hard..."):

        summary_file, text = get_text_and_summary_file(uploaded_file, url, page_range, clean_document)
        summarized_text, costs = summarize_document(text, number_of_paragraphs)

        """
        ### Summarization result
        """
        with st.expander("Cost and file download", expanded=True):
            st.text(costs)

            if is_pdf(uploaded_file):
                btn = st.download_button("Download source file plain text", text, file_name=f"{uploaded_file.name}.txt")

            st.download_button("Download summarized text", summarized_text, file_name=f"{summary_file}-summarize.md")

        st.markdown(summarized_text)