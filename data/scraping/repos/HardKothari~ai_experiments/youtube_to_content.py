import streamlit as st
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from prompt_templates_json import PROMPTS
from langsmith.run_helpers import traceable
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain.document_loaders import YoutubeLoader
from prompts_collection import summary_generator, blog_generator, tweet_generator, newsletter_generator
import asyncio

import os
from dotenv import load_dotenv

load_dotenv()

def generate_content(openai_api_key, text, content ,thread:bool=False):
    model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.5)
   

    if content == "Summary":
        prompt = summary_generator()
    elif content == "Blog":
        prompt = blog_generator()
    elif content == "Tweet":
        prompt = tweet_generator(thread=thread)
    elif content == "Newsletter":
        prompt = newsletter_generator()
    
    chain = prompt | model | StrOutputParser()

    answer = chain.invoke({"text":text})

    return answer

def main():
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key (mandatory)", type="password")

    st.header("YouTube -> Content Creator")

    yt_url = st.text_input("Enter your YouTube URL")

    summary = st.sidebar.checkbox("Summary")
    blog = st.sidebar.checkbox("Blog")
    tweet = st.sidebar.checkbox("Tweet")
    if tweet:
        thread = st.sidebar.checkbox("TweetThread")
    newsletter = st.sidebar.checkbox("Newsletter")        

    if openai_api_key and yt_url:
        if st.button("Submit"):
            # Load Transcript
            loader = YoutubeLoader.from_youtube_url(yt_url, language=["en", "en-US"])
            transcript = loader.load()

            tasks = []
            if summary or tweet:
                
                if summary:
                    summary_answer = generate_content(openai_api_key=openai_api_key, text=transcript, content="Summary")
                if tweet:
                    tweet_answer = generate_content(openai_api_key=openai_api_key, text=transcript, content="Tweet", thread=thread)
                
           
            if blog or newsletter:

                if blog:
                    blog_answer = generate_content(openai_api_key=openai_api_key, text=transcript, content="Blog")
                if newsletter:
                    newsletter_answer = generate_content(openai_api_key=openai_api_key, text=transcript, content="Newsletter")

            col1, col2 = st.columns(2)
            with col1:
                if summary:
                    st.subheader("Summary:")
                    st.write(summary_answer)
            with col2:
                if tweet:
                    st.subheader("Tweet:")
                    st.write(tweet_answer)   
            
            col1, col2 = st.columns(2)                    
            with col1:
                if blog:
                    st.subheader("Blog:")
                    st.write(blog_answer)
            with col2:
                if newsletter:
                    st.subheader("Newsletter:")
                    st.write(newsletter_answer)                      
    
        # Clear API Key
        openai_api_key = ""
    else:
        st.warning("Please populate both the OpenAI API Key and the Lazy Prompt.")


if __name__ == "__main__":
    main()