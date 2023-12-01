import csv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
import main_embed
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import openai

app = FastAPI()

def load_articles_from_csv():
    articles = []
    with open('summary_embeddings.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            summary = row['summary']
            preview = summary[:100]
            url = row['url']
            title = row['title']
            id = row['url'].split("/")[-1]  # Assuming url ends with a unique id
            articles.append({"id": id, "title": title, "summary": summary, "url": url, "full_summary": summary})
    return articles

def get_article_by_id(id: str, articles):
    for article in articles:
        if article["id"] == id:
            return article
    return None  # Return None if article not found

def my_task():
    print("Paper retrieval task initiated")
    keyword = "large AND language AND models"
    n = 60
    save_directory = "saved_articles"
    main_embed.main(keyword, n, save_directory)
    print("Paper retrieval task completed")

@app.on_event("startup")
async def startup_event():
    print("Task engine started")
    my_task()
    scheduler = BackgroundScheduler()
    scheduler.add_job(my_task, 'interval', hours=1)
    # scheduler.add_job(my_task, 'interval', seconds=30)
    scheduler.start()

@app.get("/articles/")
async def read_articles(page: Optional[int] = 1):
    articles = load_articles_from_csv()
    start = (page - 1) * 16
    end = start + 16
    paginated_articles = articles[start:end]
    return paginated_articles

@app.get("/articles/{article_id}")
async def read_article(article_id: str):
    articles = load_articles_from_csv()
    article = get_article_by_id(article_id, articles)
    if article:
        return article
    else:
        return {"error": "Article not found"}
app.mount("/", StaticFiles(directory="static", html=True), name="static")
