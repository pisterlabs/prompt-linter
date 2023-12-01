import time
import os
import openai
import json
import requests
import pyshorteners
from requests_oauthlib import OAuth1Session
from datetime import datetime, timedelta

from dotenv import load_dotenv



load_dotenv()  # Load API keys from .env file

# Get the API keys from the environment variables
CONSUMER_KEY = os.getenv("CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("CONSUMER_SECRET")
NEWS_API = os.getenv("NEWS_API")
OPENAI_API = os.getenv("OPENAI_API")

# ------------------ URL Shortening ------------------

def shorten_url(url):
    s = pyshorteners.Shortener()
    short_url = s.tinyurl.short(url)
    return short_url


# ------------------ Headline Management ------------------

def save_commented_headlines_to_file(commented_headlines):
    """Save the list of commented headlines to a JSON file."""
    with open("tokens/commented_headlines.json", "w") as f:
        json.dump(commented_headlines, f)

def load_commented_headlines_from_file():
    """Load the list of commented headlines from a JSON file."""
    try:
        with open("tokens/commented_headlines.json", "r") as f:
            commented_headlines = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print('here')
        commented_headlines = []

    return commented_headlines


def is_headline_commented(headline):
    """Check if a headline has already been commented."""
    commented_headlines = load_commented_headlines_from_file()
    return headline in commented_headlines

def save_headline_as_commented(headline):
    """Save a headline as commented by appending it to the list of commented headlines."""
    commented_headlines = load_commented_headlines_from_file()
    commented_headlines.append(headline)
    save_commented_headlines_to_file(commented_headlines)

def load_access_tokens():
    """Load access tokens from a JSON file."""
    with open("tokens/access_tokens.json", "r") as f:
        tokens = json.load(f)
    return tokens["access_token"], tokens["access_token_secret"]


def save_access_tokens(access_token, access_token_secret):
    """Save access tokens to a JSON file."""
    with open("tokens/access_tokens.json", "w") as f:
        json.dump({"access_token": access_token, "access_token_secret": access_token_secret}, f)



def authenticate(consumer_key, consumer_secret):
    """
    Authenticate with Twitter API and return access tokens.

    If access tokens are not available, perform the OAuth1.0a flow to obtain them.
    """
    access_token, access_token_secret = load_access_tokens()

    if not access_token or not access_token_secret:
        request_token_url = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
        oauth = OAuth1Session(consumer_key, client_secret=consumer_secret)

        try:
            fetch_response = oauth.fetch_request_token(request_token_url)
        except ValueError:
            print("There may have been an issue with the consumer_key or consumer_secret you entered.")

        resource_owner_key = fetch_response.get("oauth_token")
        resource_owner_secret = fetch_response.get("oauth_token_secret")

        base_authorization_url = "https://api.twitter.com/oauth/authorize"
        authorization_url = oauth.authorization_url(base_authorization_url)
        print("Please go here and authorize: %s" % authorization_url)
        verifier = input("Paste the PIN here: ")

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

        save_access_tokens(access_token, access_token_secret)

    return access_token, access_token_secret


def post_tweet(consumer_key, consumer_secret, access_token, access_token_secret, tweet_text):
    """
    Post a tweet using the Twitter API.

    Raise an exception if the request returns an error.
    """
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret,
    )

    payload = {"text": tweet_text}
    response = oauth.post("https://api.twitter.com/2/tweets", json=payload)

    if response.status_code != 201:
        raise Exception("Request returned an error: {} {}".format(response.status_code, response.text))

    print("Response code: {}".format(response.status_code))
    json_response = response.json()
    print(json.dumps(json_response, indent=4, sort_keys=True))


def get_latest_news(query, country='us', page_size=10):
    """
    Fetch the latest news articles based on the specified query and country.

    Args:
        query (str): The query to search for in the news articles.
        country (str, optional): The country to fetch news articles from. Defaults to 'us'.
        page_size (int, optional): The number of articles to fetch. Defaults to 10.

    Raises:
        Exception: If the request to the News API returns an error.

    Returns:
        list: A list of news articles as dictionaries.
    """
    api_key = NEWS_API
    
    url = "https://newsapi.org/v2/top-headlines"

    params = {
        "apiKey": api_key,
        "country": country,
        "category": "politics",
        "pageSize": page_size,
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception("Request returned an error: {} {}".format(response.status_code, response.text))

    news_data = response.json()
    return news_data["articles"]
    
def generate_comments(article, model="gpt-3.5-turbo", max_tokens=60, max_attempts=3):
    # Set the OpenAI API key
    openai.api_key = OPENAI_API

    # Initialize the number of attempts to 0
    attempt_count = 0

    # Extract the headline and URL from the article
    headline = article["title"]
    url = article["url"]
    
    # Get the shortened version of the URL
    short_url = shorten_url(url)

    # Construct the prompt for the AI model
    prompt = f"Please write a tweet about these news. The tweet must be within the allowed character length for Twitter, \
        be useful for my users, and attract users to subscribe to my Twitter account. In approximately 50% of the tweets, include a call to action \
        encouraging users to subscribe to Chatocracy. Please include the shortened URL to the source (always uncut), \
        an appropriate hashtag (only one, to keep the tweet length in check), and relevant emojis such as 📰, 🗞️, 💡, or 🚀 or other appropriate emoji \
        Write all tweets in the style of Pam Moore without repeating the headline itself in the tweet. \
        Be creative and make a meaningful comment. Include a joke if it is appropriate and causes no harm. \
        URL must present in all tweets. \
        The tweet length together with the short url must be less than 280 symbols \
        Every tweet must start from the url this is obligatory \
        Here is the headline: '{headline}', and the short url: '{short_url}'"


    # Keep generating tweets until a valid tweet is created or maximum attempts exceeded
    while attempt_count < max_attempts:

        # Call the OpenAI API to generate a tweet using the prompt
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are a very creative AI assistant that helps users write tweets in Pam Moore Style."}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # Extract the generated tweet from the response
        comment = response.choices[0].message['content'].strip()
        
        # Remove the "1. " from the beginning of the tweet if present
        comment = comment.replace("1. ", "", 1).replace("1) ", "", 1).replace("1)", "", 1).replace("1.", "", 1).replace('"', "", 1)
        
        # Check if the generated tweet is within the allowed character limit and contains the shortened URL
        if len(comment) <= 280 and short_url in comment:
            return comment
        else:
            # Increase the attempt count and reduce max_tokens to generate a shorter tweet
            attempt_count += 1
            max_tokens = max_tokens - 5
    
    # If the maximum attempts are exceeded and a valid tweet is not created, return None
    return None



if __name__ == "__main__":
    # Set the news query to "politics"
    query = "politics"
    
    # Authenticate with Twitter and get access tokens
    access_token, access_token_secret = authenticate(CONSUMER_KEY, CONSUMER_SECRET)
    # Run the script indefinitely
    while True:
        # Get the top political headlines
        top_political_headlines = get_latest_news(query)
        
        # Iterate through the headlines and their corresponding comments
        for article in top_political_headlines:
            headline = article["title"]
            
            # Check if the headline has already been commented on
            if not is_headline_commented(headline):
                # Generate comments for the headlines
                comment = generate_comments(article)
                
                # Print the headline and the generated comment
                print(f"Headline: {headline}")
                print(f"Comment: {comment}")
                print()
                
                if comment is not None:
                    #Post the tweet with the generated comment
                    tweet = post_tweet(CONSUMER_KEY, CONSUMER_SECRET, access_token, access_token_secret, comment)
                    print('!!!------!!!')
                    # Save the headline as commented
                    save_headline_as_commented(headline)
                
                # Sleep for an hour before tweeting the next comment
                time.sleep(7200)
        # Sleep for an hour before fetching the next portion of news
        time.sleep(7200)
