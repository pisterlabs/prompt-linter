import streamlit as st
import requests
from bs4 import BeautifulSoup
import random
import time
import openai

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 OPR/76.0.4017.123",
]

openai.api_key = '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'

def generate_checkbox_list(urls):
    checkbox_list = []
    for i, url in enumerate(urls):
        checkbox = st.checkbox(url, key=f"checkbox_{i}")
        checkbox_list.append(checkbox)

    return checkbox_list

def get_urls_from_webpage(url):
    delay = random.uniform(0.5, 1)  # Random delay between 0.5 and 1 second
    time.sleep(delay)

    user_agent = random.choice(USER_AGENTS)
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")
        urls = [link["href"] for link in links if link.get("href")]
        urls = [url for url in urls if url.startswith("http")]
    except requests.RequestException:
        return []

    return urls

def get_summary(url):
    delay = random.uniform(0.5, 1)  # Random delay between 0.5 and 1 second
    time.sleep(delay)

    user_agent = random.choice(USER_AGENTS)
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text().strip()
    except requests.RequestException:
        return "Error retrieving summary for {}".format(url)

    return text

def generate_summary_with_gpt(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=100,
            temperature=0.7,
            n=1,
            stop=None,

        )
        summary = response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

    return summary

def main():
    st.title("URL Select List and Summarizer")

    # URL of the webpage to scrape for URLs
    # webpage_url = "https://www.vic.ai/resources/the-must-listen-to-ai-and-ai-generative-podcasts-2023"
    webpage_url = 'https://podcasts.apple.com/us/podcast/the-ai-in-business-podcast/id670771965'

    urls = get_urls_from_webpage(webpage_url)
    if not urls:
        st.write("Error retrieving URLs from the webpage.")

    checkbox_list = generate_checkbox_list(urls)

    # You can access the checkbox values
    selected_urls = [url for url, checkbox in zip(urls, checkbox_list) if checkbox]
    st.write("Selected URLs:", selected_urls)

    # Sidebar for API Key
    st.sidebar.title("OpenAI API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

    if st.button("Create Summary"):
        openai.api_key = api_key

        for url in selected_urls:
            with st.expander("Summary for {}".format(url)):
                # Retrieve the URL content and generate summary
                text = get_summary(url)
                summary = generate_summary_with_gpt(text)
                st.write(summary)

if __name__ == "__main__":
    main()
