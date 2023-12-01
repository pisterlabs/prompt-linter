import openai
import tweepy

# Load your API key from an environment variable or secret management service
openai.api_key = "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"
_consumer_key = "pv1LOSv3B1sh24RhQOVUpFr8R"
_consumer_secret = "ZzfCkLz84F4u66i4MF6iDxi80acVUWUt5uJ4xSdjjszYZNw092"
_access_token = "1538028743796871168-CRTzRDMx7yFrF5UV08x7Yw6EkPUDcu"
_access_token_secret = "6wYIriA1neKBwUb1yWVugRZSN4m9wDSCWPeu9jXU6qEGZ"

auth = tweepy.OAuthHandler(_consumer_key, _consumer_secret)
auth.set_access_token(_access_token, _access_token_secret)

api = tweepy.API(auth)
tweets = api.search_tweets("bomb blast lang:en -is:retweet", tweet_mode='extended', count=30)

twitter_tweets = []
for tweet in tweets:  
    try:
        twitter_tweets.append(tweet.retweeted_status.full_text)
    except AttributeError:
        twitter_tweets.append(tweet.full_text)

responses = []
for i in range(len(twitter_tweets)):
    text = "Decide the sentiment of the tweet is disastrous or non-disastrous\nTweet:'" + twitter_tweets[i] + "'\nsentiment:"
    response = openai.Completion.create(model="text-davinci-002",
              prompt=text,
              temperature=0,
              max_tokens=80,
              top_p=1.0,
              frequency_penalty=0.5,
              presence_penalty=0.0)
    responses.append(response['choices'][0]['text'])

sentiment = []
for response in responses:
    text = response.lower()
    if 'non-disastrous' in text:
        sentiment.append(0)
    else:
        sentiment.append(1)

count_1 = sentiment.count(1)
count_0 = sentiment.count(0)
final_score = count_1 * 100/(count_0 + count_1) 

def nuke_func():
    return final_score