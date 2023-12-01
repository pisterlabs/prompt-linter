import tweepy
from datetime import datetime
from datetime import timedelta
import openai
import pandas as pd
import matplotlib.pyplot as plt

# Replace with your own API key and API secret key
api_key = "your key"
api_secret_key = "your secret key"
openai.api_key = "your openAi key"

# Authenticate to Twitter
auth = tweepy.OAuth1UserHandler(api_key, api_secret_key)

# Create API object
api = tweepy.API(auth)

# Create a function to handle pagination and rate limits
def get_tweets(keyword, until, count):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword,until=until).items(count):
        tweets.append(tweet)
    return tweets

# Get tweets that contain the keyword "Elon Musk" in the last 7 days
keyword = 'Elon Musk'
count = 1000

tweets = []
i=0
while i < 7:
    until = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
    tweets_store = get_tweets(keyword,until, count)
    tweets.append(tweets_store)
    i=i+1
    


chatbot = openai.Completion()
df = pd.DataFrame(columns=['Date', 'Sentiment'])

for sub_list in tweets:
    for tweet in sub_list:
        print(tweet.text)
        print(tweet.created_at.strftime('%Y-%m-%d'))
        response = chatbot.create(engine="text-babbage-001", prompt="return the most likely sentiment opinion of elon musk of the person who posted this tweet as only the word \"positive\", \"negative\", or \"neutral\" : " + tweet.text, max_tokens=20, temperature=0)
        print(response.choices[0].text.strip())
        df = df.append({'Date': tweet.created_at.strftime('%Y-%m-%d'), 'Sentiment':response.choices[0].text.strip()}
                       ,ignore_index=True)

df = df.loc[df['Sentiment'].isin(['positive','negative','neutral'])]
df['Sentiment'] = df['Sentiment'].replace({'positive':1,'negative':-1,'neutral':0})
df = df.groupby('Date').sum()
df.plot(kind='bar')
plt.show()
