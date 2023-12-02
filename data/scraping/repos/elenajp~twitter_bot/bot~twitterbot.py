#!/usr/bin/env python3
import os

import openai
import tweepy
from keys import keys  # Don't need this line in AWS Lambda
from tweepy import Cursor

API_KEY = keys["API_KEY"]
API_SECRET_KEY = keys["API_SECRET_KEY"]
ACCESS_TOKEN = keys["ACCESS_TOKEN"]
ACCESS_TOKEN_SECRET = keys["ACCESS_TOKEN_SECRET"]

openai.api_key = keys["OPENAPI_SECRET_KEY"]

auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


def retweet_comment_and_like():
    """Retweets Twitter posts containing a specific hashtag, likes the tweet and comments on it too"""
    hashtags = "#saveoursharks OR #sharkawareness OR #sharklover OR #savesharks OR #sharkdiving OR #ilovesharks OR #protectsharks"
    # Searches for tweets with a certain hashtag, language and the full text (extended) of the tweet is returned
    for tweet in Cursor(
        api.search_tweets, q=hashtags, lang="en", tweet_mode="extended"
    ).items(2):
        try:
            # Checks if the tweet has not already been retweeted, then if not, retweets it
            if not tweet.retweeted:
                api.retweet(tweet.id)
                tweet.favorite()
                status = api.get_status(tweet.id, tweet_mode="extended")
                screen_name = status.user.screen_name
                message = f"@{screen_name} Great tweet! I really enjoyed it."
                api.update_status(message, in_reply_to_status_id=tweet.id)
                print("Retweeted tweet: " + tweet.full_text)
        except Exception as error:
            print("Error: " + str(error))


def reply_to_mentions():
    """Replies to mentions with a random shark fact mentioned by chatGPT"""
    # Get the latest mention
    mentions = api.mentions_timeline()
    latest_mention = mentions[0]

    # Use OpenAI to generate a reply to the latest mention
    model_engine = "text-davinci-002"
    # prompt = "Reply to @" + latest_mention.user.screen_name + ": " + latest_mention.text
    prompt = "Mention a shark fact"

    # Load the ids of replied tweets from a file
    replied_tweet_ids = set()
    if os.path.exists("ids.txt"):
        with open("replied_tweet_ids.txt", "r") as f:
            for line in f:
                replied_tweet_ids.add(int(line.strip()))

    if latest_mention.id not in replied_tweet_ids:
        try:
            completion = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=160,
                n=1,
                stop=None,
                temperature=0.5,
            )

            reply = completion.choices[0].text
            reply = f"Thanks @{latest_mention.user.screen_name}, let me throw a shark fact at ya: {reply}"
            api.create_favorite(latest_mention.id)

            # Post the reply
            api.update_status(status=reply, in_reply_to_status_id=latest_mention.id)
            print("Successfully replied with:", reply)
            # Add the tweet id to the set of replied tweet ids
            replied_tweet_ids.add(latest_mention.id)
        except Exception as e:
            print("Error:", e)


retweet_comment_and_like()
reply_to_mentions()
