import time
import tweepy
import json
from config import (
    API_KEY,
    API_SECRET_KEY,
    BEARER_TOKEN,
    ACCESS_TOKEN,
    ACCESS_TOKEN_SECRET,
    BEN_OPENAI_API_KEY
)

from openai_bot import JokeBot

BOT_SCREEN_NAME = "JokWhySoSerious"


def get_client():
    client = tweepy.Client(
        bearer_token=BEARER_TOKEN,
        consumer_key=API_KEY,
        consumer_secret=API_SECRET_KEY,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )

    return client


def get_tweets(client: tweepy.Client, query: str):
    tweets = client.search_recent_tweets(query=query, max_results=10)
    return tweets


def main(client: tweepy.Client, bot_id: int, joke_bot: JokeBot):

    try:
        # Load the last id of the last mention that the bot replied to
        with open("last_id.json", "r") as file:
            last_id = json.load(file)["last_id"]

        while True:

            # NOTE: Projveri ima li since_id argument ovdje i kako ga iskoristiti!!!???
            mentions = client.get_users_mentions(id=bot_id, since_id=last_id)

            if mentions.data is not None:
                mention_ids = [mention.id for mention in mentions.data]
                last_id = mention_ids[0]

                # Save the last id to a file so that we can use it later if the bot crashes
                with open("last_id.json", "w") as file:
                    json.dump({'last_id': last_id}, file)

                # NOTE: The mentions are reversed so that the bot replies to the oldest mention first
                for mention in reversed(mentions.data):

                    try:
                        # print(mention.text)
                        # print(f"Meniton ID: {mention.id}\n\n")
                        prompt = f"Generate jokes based on tweets and keywords. Respond with a joke based on the words used in the following tweet: {mention.text}"
                        joke = joke_bot.generate_answer(prompt)
                        client.create_tweet(
                            text=f"{joke}", in_reply_to_tweet_id=mention.id)
                    except Exception as e:
                        print(e)

                    # VIDEO: https://www.youtube.com/watch?v=6FLeguySZLc

            else:
                # If no mentions are found, the type is tweepy.Client.Response
                print(type(mentions))
                print("No mentions.")

            # NOTE: Wait for 60 seconds before checking for new mentions becasue of the Twitter API rate limit (Basic access level)
            time.sleep(60)
            break   # TODO: Remove this in production

    except Exception as e:
        print(f"Bot chrashed at {time.ctime()}.\nError: {e}")
        with open("last_id.json", "w") as file:
            json.dump({'last_id': last_id}, file)


if __name__ == "__main__":
    client = get_client()
    user = client.get_user(username=BOT_SCREEN_NAME)
    bot_id = user.data.id
    joke_bot = JokeBot(BEN_OPENAI_API_KEY)
    main(client, bot_id, joke_bot)


# QUESTIONS:
# NOTE: Should my bot reply to direct messages?
# NOTE: How do I counter the open rate limit of 300 requests per 15 minutes and the charging???? Check out the correct numbers.

# NOTE: With Basic you get 15 requests per 15 minutes

# NOTE:
"""
Environment Variables: Since your bot requires API keys and other sensitive information, it's important to securely manage your credentials. 
Heroku provides a way to set environment variables that can be accessed by your application. This allows you to store sensitive information 
separately and ensure it's not exposed in your codebase.
"""
