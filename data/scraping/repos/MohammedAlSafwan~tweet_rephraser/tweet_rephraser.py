import re
import tweepy
import openai
import time
from config import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET, OPENAI_API_KEY, LIST_ID, TARGET_ACCOUNT, REPHRASE_OPEN_AI_ENGINE, TRANSLATE_OPEN_AI_ENGINE

# Set up Tweepy and OpenAI API
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

openai.api_key = OPENAI_API_KEY

# Function to rephrase a tweet


def rephrase_tweet(tweet):
    prompt = f"Rephrase the following tweet:\n\n{tweet}\n\nRephrased:"
    response = openai.Completion.create(
        engine=REPHRASE_OPEN_AI_ENGINE,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function to translate a tweet to Arabic


def translate_tweet_to_arabic(tweet):
    prompt = f"Translate the following English tweet into Arabic without translating URLs and emojis:\n\n{tweet}\n\nTranslation:"
    response = openai.Completion.create(
        engine=TRANSLATE_OPEN_AI_ENGINE,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function to fetch tweets from a Twitter list


def get_tweets_from_list(LIST_ID, count):
    return api.list_timeline(LIST_ID, count=count)

# Function to post a tweet


def post_tweet(tweet):
    api.update_status(tweet)

# Main function


def main():
    tweets = get_tweets_from_list(LIST_ID, 10)  # Fetch 10 tweets from the list

    for tweet in tweets:
        rephrased_tweet = rephrase_tweet(tweet.text)
        arabic_tweet = translate_tweet_to_arabic(rephrased_tweet)

        post_tweet(f"@{TARGET_ACCOUNT} {arabic_tweet}")


if __name__ == "__main__":
    while True:
        main()
        time.sleep(3600)  # Wait for an hour before restarting
