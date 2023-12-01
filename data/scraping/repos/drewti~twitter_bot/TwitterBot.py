import feedparser
import requests
import openai
import random
import re
from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1Session
import json
import time
import urllib

freq_mins = 360
use_twitter_API = True
use_bitly_API = True
debug = False
use_openAI_API = True

def shorten_url(url):
    key = 'INSERT_YOUR_CUTTLY_API_KEY'

    response = requests.get('http://cutt.ly/api/api.php?key={}&short={}'.format(key, url))
    json_data = json.loads(response.text)

    if json_data['url']['status'] == 7:
        short_link = json_data['url']['shortLink']
        print('Shortened URL:', short_link)
        return short_link
    else:
        print('Error occurred:', json_data)

def get_article_text(article_url):
    response = requests.get(article_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = ""
    for p in soup.find_all('p'):
        article_text += p.text + " "
    if debug:
        print("article len:", len(article_text.strip()))
    return article_text.strip()

def get_latest_news():
    feed_url = "https://news.google.com/rss?hl=en&gl=EN&ceid=EN:en" #"https://news.google.com/rss?hl=ru&gl=RU&ceid=RU:ru" #"https://news.google.com/rss/search?q=music&hl=en-US&gl=US&ceid=US:en"
    news_feed = feedparser.parse(feed_url)
    return news_feed['entries']

def generate_tweet(article_text, article_url):
    if use_openAI_API:
        message = " "
        while len(message) < 30 or len(message) > 220:
            openai.api_key = "INSERT_YOUR_OPENAI_API_KEY"
            model_engine = "text-davinci-002"
            #prompt = "сделай короткую (200 символов максимум) реакцию на новость в стиле Джорджа Карлина. Статья:"
            prompt = "Make short (200 symbols max) reaction on news below in the style of George Carlin suitable for twitter post with 1-2 hashtags. Article:"
            completions = openai.Completion.create(
                engine=model_engine,
                prompt=prompt + article_text,
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.7,
                stream=False
            )
            message = completions.choices[0].text
            print("choice1:", completions.choices[0].text)

            if len(message) < 30 or len(message) > 220:
                if debug:
                    print("less than 30 or more than 220 symbols answer, wait 60 seconds")
                time.sleep(60)
            else:
                message = message.strip()
                if message.startswith("George Carlin: "):
                    message = message.replace(",George Carlin: ", "", 1)

            #message = re.sub('[^0-9a-zA-Z\n\.\?! ]+', '', message).strip()
        if use_bitly_API:
            tweet = message + " " + shorten_url(article_url)
        else:
            tweet = message
        return tweet
    else:
        return "I don't know what to say"

def choose_news_topic(articles):
    article_text = " "
    while len(article_text) < 2000 or len(article_text) > 16000:
        random_number = random.randint(1, len(articles))
        article_title = articles[int(random_number)-1]['title']
        article_url = articles[int(random_number)-1]['link']
        article_text = get_article_text(article_url)
    if debug:
        print("article text chosen, len is", len(article_text))
    return article_title, article_url, article_text

def main():
    if use_twitter_API:
        consumer_key = "INSERT_YOUR_TWITTER_CONSUMER_KEY"
        consumer_secret = "INSERT_YOUR_TWITTER_CONSUMER_SECRET"
        # Get request token
        request_token_url = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
        oauth = OAuth1Session(consumer_key, client_secret=consumer_secret)
        try:
            fetch_response = oauth.fetch_request_token(request_token_url)
        except ValueError:
            print(
                "There may have been an issue with the consumer_key or consumer_secret you entered."
            )

        resource_owner_key = fetch_response.get("oauth_token")
        resource_owner_secret = fetch_response.get("oauth_token_secret")
        print("Got OAuth token: %s" % resource_owner_key)

        # Get authorization
        base_authorization_url = "https://api.twitter.com/oauth/authorize"
        authorization_url = oauth.authorization_url(base_authorization_url)
        print("Please go here and authorize: %s" % authorization_url)
        verifier = input("Paste the PIN here: ")

        # Get the access token
        access_token_url = "https://api.twitter.com/oauth/access_token"
        oauth = OAuth1Session(
            consumer_key,
            client_secret=consumer_secret,
            resource_owner_key=resource_owner_key,
            resource_owner_secret=resource_owner_secret,
            verifier=verifier,
        )
        oauth_tokens = oauth.fetch_access_token(access_token_url)

        access_token = oauth_tokens["oauth_token"]
        access_token_secret = oauth_tokens["oauth_token_secret"]

    while True:
        if debug:
            print("Starting. Let's find articles")
        articles = get_latest_news()
        if debug:
            print("articles loaded")
        topic, article_url, article_text = choose_news_topic(articles)
        if debug:
            print("topic found:", topic)
        tweet = generate_tweet(article_text, article_url)
        print(f"Generated tweet: {tweet}")
        
        # Make the request
        if use_twitter_API:
            oauth = OAuth1Session(
                consumer_key,
                client_secret=consumer_secret,
                resource_owner_key=access_token,
                resource_owner_secret=access_token_secret,
            )

            payload = {"text": tweet}

            # Making the request
            response = oauth.post(
                "https://api.twitter.com/2/tweets",
                json=payload,
            )

            if response.status_code != 201:
                raise Exception(
                    "Request returned an error: {} {}".format(response.status_code, response.text)
                )

            print("Response code: {}".format(response.status_code))

            # Saving the response as JSON
            json_response = response.json()
            print(json.dumps(json_response, indent=4, sort_keys=True))
        if debug:
            print(f"sleep for {freq_mins*60} sec")
        time.sleep(freq_mins * 60)
        if debug:
            print("wake up")

if __name__ == "__main__":
    main()
