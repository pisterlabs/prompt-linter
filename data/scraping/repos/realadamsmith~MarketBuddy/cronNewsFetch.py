import time
import os
from newsapi import NewsApiClient
import psycopg2
import openai
import requests

from newspaper import Article
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

newsapi = os.environ["NEWS_API_CLIENT_KEY"]
openai.api_key =  os.environ["OPENAI_API_KEY"]
account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)
twilio_phone_number = os.environ["TWILIO_PHONE_NUMBER"]
your_phone_number = os.environ["YOUR_PHONE_NUMBER"]

mydb = psycopg2.connect(
  host="localhost",
  user="postgres",
  password="password",
  database="postgres"
)

cursor = mydb.cursor()
    
def get_latest_publishedAt_from_db():
    cursor.execute("SELECT publishedAt FROM news ORDER BY publishedAt DESC LIMIT 1")
    result = cursor.fetchone()
    if result:
        publishedAt = result[0]
        return publishedAt
    return None

def send_sms(title, summary):
    client = Client(account_sid, auth_token)
    body = f"Title: {title}\nSummary: {summary}"
    message = client.messages.create(body=body, from_=twilio_phone_number, to=your_phone_number)
    print(body)
    print("Message Sent", message.sid)
    return message.sid
    
def fetch_and_store_news():
    latest_publishedAt = get_latest_publishedAt_from_db()
    query_params = {
        'q': 'Artificial Intelligence',
        # 'sources': 'bloomberg, techcrunch, reuters',
        'language': 'en',
        'pageSize': 4,
        'sortBy': 'publishedAt',
        'apiKey': newsapi,
    }
    
    if latest_publishedAt:
        query_params['from'] = latest_publishedAt
    url = 'https://newsapi.org/v2/everything'
    response = requests.get(url, params=query_params)

    if response.status_code == 200:
        articles = response.json()['articles']
    else:
        print(f"Error fetching articles. Status code: {response.status_code}")
        return
    
    for a in articles:
        add_article = ("INSERT INTO news "
                       "(title, author, description, url, urlToImage, publishedAt) "
                       "VALUES (%s, %s, %s, %s, %s, %s)")
        data_article = (a['title'], a['author'], a['description'], a['url'],
                        a['urlToImage'], a['publishedAt'])
        
        url = a['url']
        article = Article(url)
        try:
            article.download()
            article.parse()
            article_content = article.text        
        except Exception as e:
            print(f"Error downloading and parsing article at URL {url}: {e}")
            continue
        prompt = f"Please summarize the following article into 2 bullet points each with a max of 20 words:\n\n{article_content}\n"

        summary = openai.Completion.create(model='text-davinci-002', 
                                           prompt=prompt, 
                                           max_tokens=200)
        summary_text = summary.choices[0].text.strip()    
        print("Summary created")

        cursor.execute("INSERT INTO chatGPTnews "
                       "(title, url, summary, publishedAt) VALUES (%s, %s, %s, %s)", (data_article[0], data_article[3], summary_text, data_article[5]))
        cursor.execute(add_article, data_article)
        print("Stored")
        send_sms(data_article[0], summary_text)
        # print("Message Sent", message.sid)

    mydb.commit()

    print(f"Fetched {len(articles)} news articles at {time.strftime('%Y-%m-%d %H:%M:%S')}")


fetch_and_store_news()
