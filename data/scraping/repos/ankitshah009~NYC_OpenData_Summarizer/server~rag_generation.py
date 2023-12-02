import json
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
import pinecone
import os
from dotenv import load_dotenv
load_dotenv('.env')
pinecone.init(api_key="39e7f82b-83bf-4797-9281-0b76cb1e5b56",
              environment="us-west4-gcp-free")
index = pinecone.Index("nychackathon")
openai.api_key = os.getenv("OPENAI_API_KEY")
open_ai_api_key = os.getenv("OPENAI_API_KEY")
print('api key', open_ai_api_key)


def get_openai_embedding(text):
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {open_ai_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002",
        "encoding_format": "float"
    }
    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            embedding = json.loads(response.text)["data"][0]["embedding"]
            return embedding
        else:
            print(f"Failed to get embedding: {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def complete(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        temperature=0,
        max_tokens=550,

    )
    return response['choices'][0]['message']['content']


def answer_nyc_question(question, k=3):

    query = get_openai_embedding(question)
    results = index.query(queries=[query], top_k=k,
                          include_metadata=True, include_values=False)
    matched_articles = results['results'][0]['matches']
    try:
        page_texts = []
        websites = []
        for article in matched_articles:
            website = article['id']
            page_text = article['metadata']['page_text']
            page_texts.append(page_text)
            websites.append(website)
    except Exception as e:
        try:
            page_texts = []
            websites = []
            for article in matched_articles[0]:
                website = article['id']
                page_text = article['metadata']['page_text']
                page_texts.append(page_text)
                websites.append(website)
        except Exception as e:
            print(f"An error occurred in getting matched articles: {e}")
            return "Sorry, I don't know the answer to that."

    combined_contexts = "\n\n---\n\n".join(page_texts)

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "You are the NYC AI Assistant. Using the context information below, provide an overview of the avalable NYC services and a step by step approach that helps the user answer their question. If the question is not relevant, politely ask the user to clarify.\n\n" +
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {question}\nAnswer:"
    )

    prompt = prompt_start + combined_contexts + prompt_end
    prompt = prompt.replace('\n', ' ')
    # replace multiple whitespace with single whitespace
    prompt = ' '.join(prompt.split())
    answer = complete(prompt)

    # list the sources in a buletted list
    sources = []
    for website in websites:
        sources.append(f"- {website}")
    sources_help = "\n".join(sources)

    return answer, sources_help
