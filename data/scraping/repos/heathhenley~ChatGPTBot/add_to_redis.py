import os
import requests

from bs4 import BeautifulSoup
import numpy as np
import openai
import redis


BLOG_URL = r"https://heathhenley.github.io"

openai.api_key = os.getenv("OPENAI_API_KEY")

def add_text_to_redis(db, text, url):
    """ Add text, url, and embedding to redis db."""
    # Get the embedding for the text
    embedding = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    vector = embedding["data"][0]["embedding"]
    vector = np.array(vector).astype(np.float32).tobytes()
    post_hash = {
        "url": url,
        "content": text,
        "embedding": vector
    }
    db.hset(name=f"blog:{url}", mapping=post_hash)


def blog_to_post_urls(base_url: str) -> list[str]:
    urls = []
    page = 0
    while True:
        if page > 0:
            blog_page = f"{base_url}/page/{page}"
        else:
            blog_page = base_url
        res = None
        try:
            res = requests.get(blog_page, timeout=10) 
            if res.status_code != 200:
                break
        except Exception as e:
            print(e)
            break
        page += 1 
        soup = BeautifulSoup(res.text, 'html.parser')
        for a in soup.find_all("a"):
            if "/posts/" in a['href'] and a['href'] not in urls:
                urls.append(a['href'])
    return urls

def post_url_to_text(url: str) -> str:
    res = None
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return ""
    except Exception as e:
        print(e)
        return ""
    soup = BeautifulSoup(res.text, 'html.parser')
    text = ""
    for p in soup.find_all("section", class_="p-article__body"):
        text += p.text
    return text


def main():
    print("connecting to Redis...")
    redis_client = redis.from_url(url=os.getenv("REDIS_URL", ""), 
        encoding='utf-8',
        decode_responses=True,
        socket_timeout=30.0)
    print("checking Redis connection...")
    if not redis_client or not redis_client.ping():
        raise Exception("Redis connection failed")
    print("Connected to Redis")

    print("Crawling my blog...")
    for url in blog_to_post_urls(BLOG_URL): 
        text = post_url_to_text(url)
        print(f"Parsing: {url}")
        add_text_to_redis(redis_client, text, url)


if __name__ == "__main__":
    main()