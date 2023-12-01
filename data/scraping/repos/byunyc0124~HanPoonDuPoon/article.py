import requests
from bs4 import BeautifulSoup
import re
import openai
import time
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def fetch_info_from_openai(word, retry_limit=2):
    if word == 'ssafy' or word == 'samsung': 
        messages = [
            {"role": "system", "content": "You are a helpful assistant that speaks Korean."},
            {"role": "user", "content": word},
        ]
        openai.api_key = os.environ["OPENAI_API_KEY"]
        retries = 0
        while retries < retry_limit:  # Keep trying until successful or retry limit is reached
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=messages
                )
                info = response.choices[0].message.content
                # info = response['choices'][0]['text'].strip()
                return info
            except Exception as e:
                print(f"An error occurred: {e}")
                retries += 1
                if retries < retry_limit:
                    print(f"Retrying {retries}/{retry_limit} in 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds before retrying
                else:
                    print("Reached retry limit. Exiting.")
                    return None
    else :
        return []

def summarize_text(text, api_key, retry_limit=2):
    openai.api_key = api_key
    retries = 0
    while retries < retry_limit:  # Keep trying until successful or retry limit is reached
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt=f"Summarize the following text in a complete sentence without truncating information:\n\n{text}",
                max_tokens=280,
                temperature=0.3
            )
            summary = response['choices'][0]['text'].strip()
            return summary 
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            if retries < retry_limit:
                print(f"Retrying {retries}/{retry_limit} in 60 seconds...")
                time.sleep(60)  # Wait for 60 seconds before retrying
            else:
                print("Reached retry limit. Exiting.")
                return None

def scrape_news(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve page with status code: {response.status_code}, URL: {url}")

    soup = BeautifulSoup(response.content, 'html.parser')
    news_content_div = soup.find('div', class_='text_cont')
    if news_content_div is None:
        raise Exception("Failed to find news content div")
    paragraphs = news_content_div.find_all('p')[:10]

    news_info = ""
    for paragraph in paragraphs:
        if "▲" not in paragraph.text:
            news_info += paragraph.text + '\n'

    return news_info

def get_page_info(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    articles_div = soup.find('div', class_='board_news')
    articles = articles_div.find_all('li')

    page_info = []
    for article in articles:
        title_tag = article.find('span', class_='title ellipsis')
        image_tag = article.find('span', class_='thumb_wrap')
        link_tag = article.find('a', href=True)
        date_tag = article.find('span', class_='date')

        if title_tag and image_tag and link_tag and date_tag:
            title = title_tag.text
            image_link = re.search(r'url\((.*?)\)', image_tag['style']).group(1)
            link = link_tag['href']
            date = date_tag.text
            page_info.append({
                "articleTitle": title,
                "articleProfile": image_link,
                "articleURL": link,
                "articleDate": date
            })
    return page_info

def get_all_info(start_page, end_page):
    all_info = []
    for i in range(start_page, end_page + 1):
        url = f'https://news.samsung.com/kr/ssafy/page/{i}' if i > 1 else 'https://news.samsung.com/kr/ssafy'
        page_info = get_page_info(url)
        if page_info:
            all_info.extend(page_info)
    return all_info

# Usage
def get_article_info(companyName) :
    start_page = 1
    end_page = 2
    if companyName == 'ssafy' : 
        info = get_all_info(start_page, end_page)
        return info
    else :
        return []
