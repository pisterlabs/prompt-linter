import openai
import tweepy
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv() # load your environment credentials from a .env file

# Twitter API credentials
consumer_key = os.environ.get('consumer_key')
consumer_secret = os.environ.get('consumer_secret')
access_token = os.environ.get('access_token')
bearer = os.environ.get('bearer')
access_token_secret = os.environ.get('access_token_secret')

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret, access_token=None, access_token_secret=None)

# Create API object
api = tweepy.API(auth)

# OpenAI API Key
openai.api_key = os.environ.get('openai_api_key')

# Get tweets containing "Bitcoin"
tweets = api.search_tweets(q="Bitcoin", lang="en", count=10) #replace count int with the number of tweets you want to retrieve.

# Keep track of the count for each sentiment
positive_count = 0
neutral_count = 0
negative_count = 0
undetermined = 0

# Keep track of sentiment and date/time
sentiments = []
dates = []

# Analyze sentiment of tweets
for tweet in tweets:
    text = tweet.text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"What is the sentiment of these tweets: {text}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    ).get("choices")[0].text

    print(response)

    # Count of positive and negative tweets
    if "positive" in response or "highly positive" in response or "bullish" in response:
        sentiment = "positive"
        positive_count += 1
    elif "negative" in response or "somewhat negative" in response or "bearish" in response:
        sentiment = "negative"
        negative_count += 1
    else:
        sentiment = "undetermined"
        undetermined += 1

sentiments.append(sentiment)
dates.append(tweet.created_at)

# Calculate the percentage for each sentiment
total_count = positive_count + neutral_count + negative_count
positive_percentage = (positive_count / total_count) * 100
negative_percentage = (negative_count / total_count) * 100
undetermined_percentage = (undetermined / total_count) * 100

# Plot the percentage of each sentiment
labels = ['Positive', 'Negative', 'Undetermined']
sizes = [positive_percentage, negative_percentage, undetermined_percentage]
colors = ['green', 'red', 'grey']

plt.pie(sizes, labels=labels, colors=colors,
autopct='%1.1f%%', startangle=140)

plt.axis('equal')
plt.show()