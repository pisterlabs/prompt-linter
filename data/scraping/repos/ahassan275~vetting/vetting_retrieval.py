import aiohttp
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
import os



import streamlit as st
import requests
import os
import openai
import requests
import json

# Google Search API setup


st.set_page_config(page_title="Vetting Assistant")
st.title("Vetting Assistant")


def google_search(query):
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.environ["GOOGLE_API_KEY"],
        "cx": os.environ["GOOGLE_CSE_ID"],
        "q": query
    }
    response = requests.get(endpoint, params=params)
    results = response.json().get("items", [])
    return [result["link"] for result in results]


def get_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"Error fetching content for {url}: {e}")
        return None


def extract_insights_from_content(content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Extract insights from this content: {content[:500]}..."}
        # Using the first 500 characters for brevity
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
    )
    return response["choices"][0]["message"]["content"]


def run_conversation(question):
    messages = [{"role": "user", "content": question}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


# Streamlit UI
st.header("Vetting Assistant")
st.write("Ask any question related to the vetting process:")

# User input
question = st.text_input("Ask a question:")

if question:
    try:
        # Get AI's response to the question
        ai_response = run_conversation(question)

        # Use AI's response as a query for Google Search
        links = google_search(ai_response)

        st.write("AI's response:")
        st.write(ai_response)

        st.write("Top search results based on AI's response:")
        for link in links[:4]:  # Only consider the first 4 links
            st.write(link)

            # Fetch the content of each link
            content = get_page_content(link)
            if content:
                # Extract insights from the content using OpenAI
                insights = extract_insights_from_content(content)
                st.write("Extracted Insights:")
                st.write(insights)

    except Exception as e:
        st.error(f"An error occurred: {e}")
