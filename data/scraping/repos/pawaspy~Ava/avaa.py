import streamlit as st
import openai
from googlesearch import search
import requests
from bs4 import BeautifulSoup

# Define API key for OpenAI
openai.api_key = "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"


# Streamlit app title
st.title("🤖Ava(Artificial Virtual Assistant)")

# Input question
question = st.text_input("Enter your question:")

if question:
    # Search Google for the top 5 results
    search_results = search(question, num_results=5)

    # Extract title and snippet from search results
    def extract_title_and_snippet(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.string if soup.title else ""
            snippet = soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else ""
            return title, snippet
        except Exception as e:
            return "", ""

    context = ""
    for url in search_results:
        title, snippet = extract_title_and_snippet(url)
        context += title + " " + snippet + " "

    # Get the answer from GPT-3
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"{question}\n{context}\nAnswer:",
        temperature=0.9,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
    )

    answer = response.choices[0].text.strip()

    # Display the answer
    st.write(answer)
