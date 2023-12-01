import requests
from bs4 import BeautifulSoup, NavigableString
import urllib.parse
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
import openai
import os
import pandas as pd
import modal
import time
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.environ['OPENAI_API_KEY']
stub = modal.Stub(image=modal.Image.debian_slim().pip_install("openai").pip_install("langchain").pip_install("pandas").pip_install("bs4"))
volume = modal.NetworkFileSystem.persisted("job-storage-vol")


MODEL_DIR = "/data"

def get_doc_urls(url):
    from bs4 import BeautifulSoup, NavigableString
    urls = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and not href.startswith('http'):
            href = urllib.parse.urljoin(url, href)
        if '/docs' in href:
            urls.append(href)
    return urls

def all_urls():
    guide_urls = get_doc_urls('https://modal.com/docs/guide')
    guide_urls.remove('https://modal.com/docs')
    guide_urls.remove('https://modal.com/docs/guide#introduction-to-modal')
    guide_urls.remove('https://modal.com/docs/guide#features')
    guide_urls.remove('https://modal.com/docs/guide#getting-started')
    guide_urls.remove('https://modal.com/docs/guide#how-does-it-work')
    ref_urls = get_doc_urls('https://modal.com/docs/reference')
    ref_urls.remove('https://modal.com/docs/reference#api-reference')
    ref_urls.remove('https://modal.com/docs')
    doc_urls = list(set(guide_urls + ref_urls))
    return doc_urls

def scrape_page(url, all_docs):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    h1_header = soup.find('h1')
    # print(f'H1: {h1_header.text.strip()}')

    # Exclude h2 headers with a specific class
    h2_headers = [h2 for h2 in soup.find_all('h2') 
                  if h2.get('class') != ['text-white', 'text-sm', 'py-2', 'pl-4', 'border-l', 'border-white/10'] 
                  and 'Try this on Modal!' not in h2.text]
    for i, h2_header in enumerate(h2_headers):
        if i == 0:  # For the first h2, get text between h1 and h2
            sibling = h1_header.find_next()
            text_content = ""
            while sibling and sibling != h2_header:
                text_content += sibling.get_text(strip=True) + " "
                sibling = sibling.find_next()
            # print(text_content)
            curr_doc = Document(page_content=text_content, metadata={"source": str(h1_header.text.strip())})
            all_docs.append(curr_doc)
            # print(f'H2: {h2_header.text.strip()}')
        else:  # For subsequent h2s, get text between this h2 and the next one
            sibling = h2_header.find_next()
            text_content = ""
            while sibling and (sibling.name != 'h2' and sibling.name != 'h1'):
                text_content += sibling.get_text(strip=True) + " "
                sibling = sibling.find_next()
            # print(text_content)
            curr_doc = Document(page_content=text_content, metadata={"source": str(h1_header.text.strip()) + " " + str(h2_header.text.strip())})
            all_docs.append(curr_doc)
            # print(f'H2: {h2_header.text.strip()}')
    last_h2 = h2_headers[-1] if h2_headers else None
    if last_h2:
        sibling = last_h2.find_next()
        text_content = ""
        while sibling:
            text_content += sibling.get_text(strip=True) + " "
            sibling = sibling.find_next()
        curr_doc = Document(page_content=text_content, metadata={"source": str(h1_header.text.strip()) + " " + str(h2_header.text.strip())})
        all_docs.append(curr_doc)
    return all_docs
        # print(text_content)

@stub.function(secret=modal.Secret.from_name("my-openai-secret"), timeout=3600*3)
def think_questions(page_content):
    import time
    import openai
    import random
    time.sleep(random.randint(0, 10))
    think_prompt = "Come up with 5 potential questions that the following documentation content could answer. Make it specific to the content presented. Present each question on a different line. Don't number the questions. \n"
    page_primer = "\nContent: \n"
    prompt = think_prompt + page_primer + page_content + "\nAnswer:\n"

    tries = 20 

    while tries > 0:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that comes up with questions and answer pairings on given content."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            tries -= 1
            time.sleep(random.randint(0, 10))
    
    if tries == 0:
        return "Error"

@stub.function(timeout=3600*3, concurrency_limit = 5)
def load_data_doc(doc):
    dataset = pd.DataFrame(columns=['question', 'topic', 'content', 'answer'])
    content = doc.page_content
    topic = doc.metadata["source"]
    answer = think_questions.remote(topic + ": " + content)
    if answer == "Error":
        print(topic + " failed")
        return dataset

    questions = answer.split("\n")
    
    for i in range(0, len(questions)):
        new_row = {'question': questions[i], 'topic' : topic, 'content' : content, 'answer' : questions[i+1]}
        dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)
    return dataset


def load_data(all_docs):
    df_list = list(load_data_doc.map(all_docs))
    dataset = pd.concat(df_list)
    return dataset

def get_docs(doc_urls):
    docs = []
    for url in doc_urls:
        docs.append(scrape_page(url, docs))
    return docs

@stub.local_entrypoint()
def main():
    doc_urls = all_urls()
    all_docs = get_docs(doc_urls)
    dataset = load_data(all_docs)
    dataset.to_csv('dataset.csv')