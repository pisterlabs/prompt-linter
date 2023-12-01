import streamlit as st
import requests
import json
import pandas as pd
import openai

openai.api_type = "azure"
openai.api_base = st.secrets.aoai.base
openai.api_key = st.secrets.aoai.key
openai.api_version = st.secrets.aoai.previewversion

def bing_web_search(subscription_key, bing_subscription_url, query,site=None):
    # set parameters
    # search_url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {
        "q": query,
        "textDecorations": True,
        "textFormat": "Raw",
        "mkt":"en-us",
        "responseFilter":"Webpages"}
    if site is not None:
        params["q"]+=" site:"+site
    
    # get response
    response = requests.get(bing_subscription_url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

def runQuery():
    site=None
    if "Specific" == st.session_state.rbSelect:
        site=st.session_state.txtUrl

    data = bing_web_search(st.secrets.bing.key, st.secrets.bing.endpoint, st.session_state.txtSearch,site=site)

    if 'webPages' in data:
        df = pd.json_normalize(data['webPages']['value'])

        prompt="""
        Answer ONLY with the facts listed in the list of sources below
        If there isn't enough information below, say you don't know.
        Do not generate answers that don't use the sources below.
        If asking a clarifying question to the user would help, ask the question.
        Each source has a URL followed by colon and the actual information, always include the source URL for each fact you use in the response.

        Return the response in markdown with the URL references
        
        Sources:
        """
        srch=[]
        for _,row in df.iterrows():
            srch.append(f"{row['url']}: {row['snippet']}")
        prompt+="\n".join(srch)
        
        messages=[]
        messages.append({"role":"system","content":prompt})
        messages.append({"role":"user","content":st.session_state.txtSearch})
        

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo", 
            messages = messages,
            temperature=0,
            max_tokens=1500
        )
        st.markdown("## Question")
        st.markdown(f"{st.session_state.txtSearch}")
        st.markdown("## Answer")
        st.markdown(f"{response['choices'][0]['message']['content']}")

        with st.expander("See more info"):
            st.write(df[['url','snippet']])
            st.write(response)
            st.write(messages)
    else:
        st.write("No results found")
        
with st.sidebar:
    rb=st.radio("Are you using any web site or specific?",("All","Specific"),key="rbSelect")
    with st.form("bingit"):
        if "Specific" == st.session_state.rbSelect:
            st.text_input("Search using this website",key="txtUrl",value="learn.microsoft.com/en-us/azure")
        st.text_input("Enter search string",value="How much is power bi",key="txtSearch")
        st.form_submit_button(label='Submit', on_click=runQuery)