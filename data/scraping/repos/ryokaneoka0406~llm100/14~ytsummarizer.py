import os

import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


def load_youtube_transcript(url: str):
    return YoutubeLoader.from_youtube_url(url)


def stuff(docs):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(docs)


def map_reduce(split_docs):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
    map_template = """The following is a set of documents
        {text}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""
    MAP_PROMPT = PromptTemplate(
        template=map_template, input_variables=["text"])
    reduce_template = """The following is set of summaries:
        {text}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Helpful Answer:"""
    REDUCE_PROMPT = PromptTemplate(
        template=reduce_template, input_variables=["text"])
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        verbose=True,
        map_prompt=MAP_PROMPT,
        combine_prompt=REDUCE_PROMPT
    )
    return chain.run(split_docs)


def refine(split_docs):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
    chain = load_summarize_chain(llm, chain_type="refine")
    return chain.run(split_docs)


def text_splitter(loader):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000,
        chunk_overlap=0
    )
    return loader.load_and_split(text_splitter=text_splitter)


st.title('YouTube Summarizer')

# env variables
user_api_key = st.sidebar.text_input(
    label="OpenAI API key",
    placeholder="Paste your OpenAI API key here",
    type="password")
os.environ['OPENAI_API_KEY'] = user_api_key

url = st.text_input("YouTube URL")
chain_type = st.selectbox("summarize type", ["stuff", "map_reduce", "refine"])
st.markdown("""
summarization type
- stuff: Appropriate for short videos.It takes a list of documents, inserts them all into a prompt and passes that prompt to an LLM.
    - [LangChain document: stuff](https://python.langchain.com/docs/modules/chains/document/stuff)
- map_reduce: It can work with long videos, but it takes time to process them.The map reduce documents chain first applies an LLM chain to each document individually (the Map step), treating the chain output as a new document. 
    - [LangChain document: map reduce](https://python.langchain.com/docs/modules/chains/document/map_reduce)
- refine: It can work with long videos, but it takes time to process them.The Refine documents chain constructs a response by looping over the input documents and iteratively updating its answer.
    - [LangChain document: refine](https://python.langchain.com/docs/modules/chains/document/refine)
""")

if st.button("Summarize", type="primary"):
    with st.spinner("Loading..."):
        if chain_type == "stuff":
            try:
                docs = load_youtube_transcript(url).load()
            except:
                st.error("Error occurreed: try other summarize type")
            st.success(stuff(docs))
        elif chain_type == "map_reduce":
            loader = load_youtube_transcript(url)
            split_docs = text_splitter(loader)
            st.success(map_reduce(split_docs))
        elif chain_type == "refine":
            loader = load_youtube_transcript(url)
            split_docs = text_splitter(loader)
            st.success(refine(split_docs))
