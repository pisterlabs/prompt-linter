import streamlit as st
import openai
from bs4 import BeautifulSoup
import requests 
from newspaper import Article
from helpers.get_links import get_article_urls
st.set_page_config(layout="wide", page_title="News Summarizer", page_icon="📰")

from st_pages import show_pages_from_config
show_pages_from_config()

# api_key = st.sidebar.text_input("OpenAI API Key", type="password")

openai.api_base = "https://api.openai.com/v1"
api_key = st.secrets["openai_api_key"]
openai.api_key = api_key
model = "gpt-3.5-turbo"

def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

st.sidebar.header("📰 News Summarizer")
news_source = st.sidebar.selectbox("**Select news source**", ("Rappler", "The Sydney Morning Herald", "Special Broadcasting Service", "Outsource Accelerator"), on_change=clear_session_state)
if news_source == "Rappler":
    category = st.sidebar.radio("Category", ("National", "Metro Manila", "Weather", "Environment"), on_change=clear_session_state)
elif news_source == "The Sydney Morning Herald":
    category = st.sidebar.radio("Category", ("Companies", "Market"), on_change=clear_session_state)
elif news_source == "Special Broadcasting Service":
    category = st.sidebar.radio("Category", ("Top Stories", "Life"), on_change=clear_session_state)
elif news_source == "Outsource Accelerator":
    category = st.sidebar.radio("Category", ("BPO News", "BPO Articles"), on_change=clear_session_state)

scrape = st.sidebar.button("Get latest news")

article_title = []
article_content = []
st.session_state.disabled = True

if scrape:
    with st.spinner("Fetching latest news..."):
        article_urls = get_article_urls(category)

        for i, url in enumerate(article_urls):
            article = Article(url)
            article.download()
            article.parse() 
            article_title.append(article.title)
            if article.text.startswith("This is AI generated summarization"):
                article_content.append(article.text[105:])
            else:
                article_content.append(article.text)
            st.session_state.article_title = article_title
            st.session_state.article_content = article_content
    st.session_state.disabled = False
        

summary_button = st.sidebar.button("Summarize Articles", disabled=st.session_state.disabled)
def generate_summary(article_content, content="You are a news summary bot. Given the content of an article, your task is to summarize it concisely into 50 words maximum."):
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{content}"},
            {"role": "user", "content": f"Content: {article_content}"}
        ]   
    )
    content = chat_completion['choices'][0]['message']['content']
    return content

col1, col2, = st.columns(2, gap="medium")

with col1:
    st.header("News Content")
    st.divider()
    if 'article_title' and 'article_content' in st.session_state:
        for i, content in enumerate(st.session_state.article_content):
            st.subheader(st.session_state.article_title[i])
            st.write(content)
            st.write("---")
with col2:
    st.header("Summarized Content")
    st.divider()
    if summary_button and 'article_title' and 'article_content' in st.session_state:
        if api_key:
            for i, content in enumerate(st.session_state.article_content):
                st.subheader(st.session_state.article_title[i])
                with st.spinner("Generating summary..."):
                    summary = generate_summary(st.session_state.article_content[i])
                    st.success(summary)
                    st.write("---")
        else:
            st.sidebar.error("Please enter your API key.")