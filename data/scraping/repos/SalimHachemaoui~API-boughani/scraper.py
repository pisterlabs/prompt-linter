import os
import requests
from bs4 import BeautifulSoup
import openai

# Initialisation du client OpenAI avec la clé API depuis une variable d'environnement
client = openai.api_key = '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'

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = soup.get_text()
        return page_content
    else:
        return None

def generate_summary(text_content):
    messages = [{"role": "user", "content": f"Résumez le texte suivant:\n{text_content}"}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
    )

    choices = response.choices
    if choices:
        summary = choices[0].message.content
        return summary
    else:
        return None
