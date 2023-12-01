import os

from langchain import OpenAI
from langchain.document_loaders import YoutubeLoader, WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

def document_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 100,
    )
    splitted_docs = text_splitter.split_documents(docs)

    return splitted_docs

def get_summary(splitted_docs):
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    summary =chain.run(splitted_docs)

    return summary

def youtube_video_summariser(url):
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()

    splitted_docs = document_splitter(docs)

    summary = get_summary(splitted_docs)
    return summary

def article_summariser(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitted_docs = document_splitter(docs)
    
    summary = get_summary(splitted_docs)
    return summary