import os
import sys
import toml
import requests
import openai
import streamlit as st
import concurrent.futures
from googlesearch import search
import time

if not os.path.exists("secrets.toml"):
    # Set API keys and model
    open_ai_api_key = st.secrets.api_keys["OPENAI_API_KEY"]
    browserless_api_key = st.secrets.api_keys["BROWSERLESS_API_KEY"]
else:
    secrets = toml.load("secrets.toml")["api_keys"]
    open_ai_api_key = secrets["OPENAI_API_KEY"]
    browserless_api_key = secrets["BROWSERLESS_API_KEY"]

openai_model = "gpt-3.5-turbo"

openai.api_key = open_ai_api_key
headers = {'Cache-Control': 'no-cache', 'Content-Type': 'application/json'}
params = {'token': browserless_api_key}

@st.cache_data
def scrape(link):
    """Scrape the content of a webpage."""
    json_data = {
        'url': link,
        'elements': [{'selector': 'body'}],
    }
    response = requests.post('https://chrome.browserless.io/scrape', params=params, headers=headers, json=json_data)

    if response.status_code == 200:
        webpage_text = response.json()['data'][0]['results'][0]['text']
        return webpage_text
    else:
        print(f"Error: Unable to fetch content from {link}. Status code: {response.status_code}")
        return ""


def summarize(question, webpage_text):
    """Summarize the relevant information from a body of text related to a question."""
    prompt = f"""You are an intelligent summarization engine. Extract and summarize the
  most relevant information from a body of text related to a question.

  Question: {question}

  Body of text to extract and summarize information from:
  {webpage_text[0:2500]}

  Relevant information:"""

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],temperature = 0.2
            )
            return response.choices[0].message.content
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Sleeping for 3 seconds.")
            time.sleep(3)

def final_summary(question, summaries):
    """Construct a final summary from a list of summaries."""
    num_summaries = len(summaries)
    prompt = f"You are an intelligent summarization engine. Extract and summarize relevant information from the {num_summaries} points below to construct an answer to a question.\n\nQuestion: {question}\n\nRelevant Information:"

    for i, summary in enumerate(summaries):
        prompt += f"\n{i + 1}. {summary}"

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Sleeping for 3 seconds.")
            time.sleep(3)

def link(r):
    """Extract the link from a search result."""
    return r['link']

@st.cache_data
def search_results(question):
    """Get search results for a question."""
    organic_results = []
    for link in search(question,tld="co.in", num=10, stop=10, pause=2):
        organic_results.append(link)
    return organic_results

def print_citations(links, summaries):
    """Print citations for the summaries."""
    st.write("CITATIONS")
    num_citations = min(len(links), len(summaries))
    for i in range(num_citations):
        st.write(f"[{i + 1}] {links[i]}\n{summaries[i]}\n")

def scrape_and_summarize(link, question):
    """Scrape the content of a webpage and summarize it."""
    webpage_text = scrape(link)
    summary = summarize(question, webpage_text)
    return summary

def main():
    st.title("CustomGPT")
    question = st.text_input("What would you like me to search?")
    if st.button("Search"):
        links = search_results(question)[:7]  # Limit the number of search results
        summaries = []

        step_placeholder = st.empty()

        with st.spinner("Learning..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_summary = {executor.submit(scrape_and_summarize, link, question): link for link in links}
                for i, future in enumerate(concurrent.futures.as_completed(future_to_summary)):
                    summaries.append(future.result())
                    step_placeholder.text(f"Step {i+1}: Scraping and summarizing link {i+1}")

            step_placeholder.text("Step 9: Generating final summary")
            answer = final_summary(question, summaries)

        step_placeholder.empty()

        st.write("HERE IS THE ANSWER")
        st.write(answer)
        print_citations(links, summaries)

if __name__ == "__main__":
    main()
