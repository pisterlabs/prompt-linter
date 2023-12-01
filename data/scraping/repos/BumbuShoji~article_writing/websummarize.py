import os
import requests
from bs4 import BeautifulSoup
import openai

# Make sure to replace 'your_openai_api_key' with your actual API key
openai.api_key = "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"

def fetch_html_content(url):
    response = requests.get(url)
    html_content = response.text
    return html_content

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup

def extract_relevant_content(soup):
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

def summarize_text(text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Please summarize the following text:\n\n{text}\n",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    summary = response.choices[0].text.strip()
    return summary

def main():
    url = input("Enter the URL: ")
    html_content = fetch_html_content(url)
    soup = parse_html(html_content)
    text = extract_relevant_content(soup)
    summary = summarize_text(text)
    print("Summary:")
    print(summary)

if __name__ == "__main__":
    main()
