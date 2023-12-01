import os
import openai
import tweepy
import logging
import random
import time
from typing import Optional

# Setup logging to write to a file
logging.basicConfig(filename='cybersecurity_tips_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Load environment variables
openai_api_key: str = os.getenv('SOCK_PUPPET_OPENAI_API_KEY')
twitter_api_key: str = os.getenv('SOCK_PUPPET_TWITTER_API_KEY')
twitter_api_secret_key: str = os.getenv('SOCK_PUPPET_TWITTER_API_SECRET')
twitter_access_token: str = os.getenv('SOCK_PUPPET_TWITTER_ACCESS_TOKEN')
twitter_access_token_secret: str = os.getenv('SOCK_PUPPET_TWITTER_ACCESS_TOKEN_SECRET')
twitter_bearer_token: str = os.getenv('SOCK_PUPPET_TWITTER_BEARER_TOKEN')

# Initialize OpenAI
openai.api_key = openai_api_key


def get_tip() -> Optional[str]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a human knowledgeable in "
                                              "cybersecurity, programming and AI."},
                {"role": "user", "content": "Can you give me a tweet-length "
                                            "cybersecurity, programming or AI tip "
                                            "(or trivia)? It can also be a pun."}
            ]
        )
        # Remember to remove the double quotes from the response!
        return response.choices[0].message['content'].replace('"', '')
    except Exception as e:
        logging.error(f"Error in fetching cybersecurity tip: {e}")
        return None


def post_to_twitter(message: str) -> None:
    try:
        client = tweepy.Client(
            consumer_key=twitter_api_key,
            consumer_secret=twitter_api_secret_key,
            access_token=twitter_access_token,
            access_token_secret=twitter_access_token_secret,
            bearer_token=twitter_bearer_token
        )
        client.create_tweet(text=message)
        logging.info(f"Cybersecurity tip posted on Twitter: {message}")
    except Exception as e:
        logging.error(f"Error in posting to Twitter: {e}")


if __name__ == "__main__":
    if random.randint(1, 100) == 1:
        tip = get_tip()

        if tip:
            delay_minutes = random.randint(0, 60)
            logging.info(f"Waiting for {delay_minutes} minutes before posting")
            time.sleep(delay_minutes * 60)  # Convert minutes to seconds
            post_to_twitter(tip)
        else:
            logging.info("No cybersecurity tip was posted.")
    else:
        logging.info("Script did not run this time.")
