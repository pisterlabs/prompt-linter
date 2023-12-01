import streamlit as st
from decouple import config

import os
from urllib.parse import unquote

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import textwrap
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from util import count_tokens_run

st.markdown("# Chat with a PDF or Web Page")

os.environ["OPENAI_API_KEY"] = config('OPENAI_API_KEY')
st.session_state.setdefault('past', [])
st.session_state.setdefault('generated', [])
llm = OpenAI(temperature=0)


def on_clear_msg_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

@st.cache_data
def decode_website(url):
    # Get the page
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
    response = requests.get(url, headers=headers)
    # Scrape the content
    soup = BeautifulSoup(response.text)
    main_tag = soup.find('main')
    main_text = main_tag.get_text(separator=' ', strip=True)
    return main_text


@st.cache_data
def summarize_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(text)
    embeddings = OpenAIEmbeddings()
    # Create a vectorstore from documents
    st.session_state.retriever = Chroma.from_documents(docs, embeddings)

    texts = text_splitter.split_text(text)
    print(len(texts))
    docs = [Document(page_content=t) for t in texts[:4]]
    chain = load_summarize_chain(llm,
                                 chain_type="map_reduce")

    output_summary = count_tokens_run(chain, docs)

    wrapped_text = textwrap.fill(output_summary, width=100)

    return wrapped_text


@st.cache_data
def extract_file_content():
    pdf_reader = PdfReader(uploaded_file)
    text = []
    for page in pdf_reader.pages:
        text.append(page.extract_text())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(text)
    embeddings = OpenAIEmbeddings()
    st.session_state.retriever = Chroma.from_documents(docs, embeddings)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return count_tokens_run(chain, docs)


genre = st.radio(
    "Chat with",
    ('Doc', 'Web Page'))

if genre == 'Doc':
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
    if uploaded_file is not None:
        st.write(extract_file_content())
else:
    query = st.text_input("Input url :")
    if query:
        encode_url = unquote(unquote(query))
        decoded_text = decode_website(encode_url)
        st.write(summarize_text(decoded_text))


def generate_response(query_text):
    if st.session_state.retriever:
        with st.spinner('Answering...'):
            docs = st.session_state.retriever.similarity_search(query=query_text, k=3)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            qa_result = chain.run(input_documents=docs, question=query_text)
            return qa_result


def on_input_change():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.past.append(user_input)
        response = generate_response(user_input)
        st.session_state.generated.append(response)
        st.session_state.user_input = ""


chat_placeholder = st.empty()
with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])):
        with st.chat_message("user"):
            st.write(st.session_state['past'][i])
        with st.chat_message("bot"):
            st.write(st.session_state['generated'][i])

st.text_input("Enter a short question:", placeholder='Please provide a short question.',
              on_change=on_input_change,
              key="user_input")

if len(st.session_state['generated']) > 0:
    st.button("Clear message", on_click=on_clear_msg_click)
