#pip install youtube-transcript-api, langchain, tiktoken, streamlit

from langchain.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain 
from langchain.text_splitter import CharacterTextSplitter

import streamlit as st
# Display the Page Title
st.title('🦜️🔗 Youtube Summarizer :red[Toolbox]')

import os
api_key = st.text_input("Paste the OpenAI API Key")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    llm = OpenAI(temperature=0) #Temp controls the randomness of the text

    prompt = st.text_input("Paste the URL")

    if prompt:
        loader = YoutubeLoader.from_youtube_url(prompt, add_video_info=False)

        docs = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, separator = " ", chunk_overlap=50)

        split_docs = text_splitter.split_documents(docs)

        chain = load_summarize_chain(llm, chain_type="map_reduce")
        
        st.write(chain.run(split_docs))

