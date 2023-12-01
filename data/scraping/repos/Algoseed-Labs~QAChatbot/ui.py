import streamlit as st
import re
import pickle
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import VectorDBQAWithSourcesChain
import argparse
import pickle
import requests
import xmltodict

from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

def Find(string):
 
    # findall() has been used
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]

def scrap_site(url):  

    r = requests.get(url)
    xml = r.text
    print(xml)
    raw = xmltodict.parse(xml)

    pages = []
    for info in raw['urlset']['url']:
        # info example: {'loc': 'https://www.paepper.com/...', 'lastmod': '2021-12-28'}
        url = info['loc']
        pages.append({'text': extract_text_from(url), 'source': url})

    text_splitter = CharacterTextSplitter(chunk_size=750, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)

_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """ You are a AI tool build to accept a sitemap.xml url in order to respond to questions regarding the site you are tasked to scrape
remind the user of this if you are unsure of a response. 
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0.35)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain



# App title
st.set_page_config(page_title="DOFINITY.AI")

# Hugging Face Credentials
with st.sidebar:
    st.title('DOFINITY.AI')
    
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Please Enter A Sitemap.xml url so i can scrape the site and respond properly :) "}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input):
    # Hugging Face Login
    # Create ChatBot               
    with open("faiss_store.pkl", "rb") as f:
         store = pickle.load(f)

    chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0.5, verbose=True), vectorstore=store, verbose=True)
    result = chain({"question": prompt_input})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

        # print(f"AI: {}")         
    return result['answer']

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generatae a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            urls=Find(prompt)
            print(urls)
            if len(urls)>0:
               print("url") 
               scrap_site(urls[0])
               st.write("Data Fetched from "+urls[0]) 
               message = {"role": "assistant", "content":"Data Fetched from "+urls[0]}
               st.session_state.messages.append(message)
               
            else:
                print("url not")

                response = generate_response(prompt) 
                st.write(response) 
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
