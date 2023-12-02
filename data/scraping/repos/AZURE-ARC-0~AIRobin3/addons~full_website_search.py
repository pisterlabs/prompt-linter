import requests
from bs4 import BeautifulSoup

description = "Used to answer questions about an entire website, not just a single page. The website is scraped and vectorized, and then the 3 most similar chunks of text are retrieved."

parameters = {
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "The URL of the website to visit.",
    },
    "query": {
      "type": "string",
      "description": "The query to retrieve the most similar chunks of text from a vectorized representation of the website.",
    },
    "include_links": {
      "type": "boolean",
      "description": "Whether or not to include links in the scraped data.",
      "default": True
    },
    "include_images": {
      "type": "boolean",
      "description": "Whether or not to include images in the scraped data.",
      "default": True
    },
  },
  "required": ["url", "query"],
}

import os
import joblib
import openai
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import re
import nltk
from tenacity import retry, wait_random_exponential, stop_after_attempt
import config

# Set the OpenAI API key
openai.api_key = config.OPENAI_API_KEY

visited_links = set()
all_links = set()

def get_website_name(url):
    return url.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]

def visit_website(url, include_links=True, include_images=True):
    
    print(f'Visiting {url}\n')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
    driver.get(url)

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    for script in soup(["script", "style"]):
        script.decompose()

    data = ''

    if include_images:
        images = [img.get('src') for img in soup.find_all('img') if img.get('src') and img.get('src').startswith('http')]
        data += '\nImages: ' + '\n'.join(images) + '\n'

    text = ' '.join(soup.stripped_strings)
    data += '\nText: ' + text

    if include_links:
        links = [link.get('href') for link in soup.find_all('a') if link.get('href') and link.get('href').startswith('http')]
        for link in links:
            if link not in all_links:
                all_links.add(link)
            if link not in visited_links and get_website_name(url) in link:
                if not re.search(r'\.(jpg|jpeg|png|gif|svg|mp4|mp3|avi|wav|mov|flv|wmv|webm|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|7z|gz|tar|iso|exe|dmg|apk|csv|tsv|json|xml|txt|rtf|odt|ods|odp|odg|odf|odb|odc|odm|ott|ots|otp|otg|otf|oti|oth|sxw|stw|sxc|stc|sxi|sti|sxd|std)', link):
                    visited_links.add(link)
                    link_data, link_url = visit_website(link)
                    data += link_data

    driver.quit()
    return data, url

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def retrieve_content(query: str, embeddings, chunks, top_n=3):
    query_embedding = get_embedding(query)
    query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

    embeddings_np = np.array(embeddings).astype('float32')

    faiss.normalize_L2(embeddings_np)
    faiss.normalize_L2(query_embedding_np)

    index = faiss.IndexFlatIP(len(query_embedding_np[0]))
    index.add(embeddings_np)
    D, I = index.search(query_embedding_np, top_n)

    most_similar_indices = I[0]
    scores = D[0]

    results = []
    for idx, score in zip(most_similar_indices, scores):
        results.append((chunks[idx], score, idx))

    return results

def process_website(website_url, include_links=True, include_images=True):
    website_name = get_website_name(website_url)

    try:
        scraped_data, url = joblib.load(f'scraped/{website_name}_scraped_data.joblib')
    except FileNotFoundError:
        scraped_data, url = visit_website(website_url, include_links, include_images)
        scraped_data += '\nLinks: ' + '\n'.join(all_links) + '\n'
        # make sure the scraped folder exists
        if not os.path.exists('scraped'):
            os.makedirs('scraped')
        joblib.dump((scraped_data, url), f'scraped/{website_name}_scraped_data.joblib')

    try:
        embeddings = joblib.load(f'scraped/{website_name}_embeddings.joblib')
        chunks = joblib.load(f'scraped/{website_name}_chunks.joblib')
    except FileNotFoundError:
        embeddings = []
        chunks = []
        content = scraped_data
        sentence_chunks = nltk.sent_tokenize(content)
        for i, sentence in enumerate(sentence_chunks):
            if i % 5 == 0:
                chunks.append({"text": sentence, "url": url})
            else:
                chunks[-1]["text"] += '\n' + sentence

        for chunk in chunks:
            embeddings.append(get_embedding(chunk["text"]))

        joblib.dump(embeddings, f'scraped/{website_name}_embeddings.joblib')
        joblib.dump(chunks, f'scraped/{website_name}_chunks.joblib')

    return embeddings, chunks

def full_website_search(url, query, include_links=True, include_images=True):
    results = []
    website_url = url
    embeddings, chunks = process_website(website_url, include_links, include_images)
    results = retrieve_content(query, embeddings, chunks)
    for i, (result, score, index) in enumerate(results):
        results[i] = f"Result {i+1} (Score: {score}):\n{result['text']}\nSource: {result['url']}\nIndex: {index}"
    return '\n\n'.join(results) + '\n\n'