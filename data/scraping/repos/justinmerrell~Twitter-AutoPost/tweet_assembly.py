import os
import sys
import json
import logging

import openai
from dotenv import load_dotenv

from .tweet_format import get_disclaimer, structure
from .tweet_concept import generate_concept


load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# ------------------------------- Setup Logger ------------------------------- #
logger = logging.getLogger("tweet_builder")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def add_disclaimer(tweet_text):
    '''
    Add the disclaimer to the end of the tweet
    '''
    disclaimer = get_disclaimer()
    return tweet_text + disclaimer


def validate_tweet(tweet_text):
    if len(tweet_text) > 280:
        logger.error("Tweet is too long. Tweet length: %i", len(tweet_text))
        return False

    return tweet_text


def get_tweet(max_attempts=10):
    '''
    Returns a tweet that is less than 280 characters.
    '''
    attempts = 0

    while attempts < max_attempts:
        logger.info("Attempt %i of %i to generate a tweet.",
                    attempts + 1, max_attempts)

        logger.info("Generating concept...")
        concept, _ = generate_concept()

        logger.info("Structuring tweet...")
        tweet, _ = structure(concept)

        logger.info("Adding disclaimer...")
        tweet = add_disclaimer(tweet)

        logger.debug("Tweet: %s", tweet)

        logger.info("Validating tweet...")
        if validate_tweet(tweet):
            # Add tweet to tweet history
            with open("tweet_history.json", "r", encoding="UTF-8") as tweet_history_file:
                tweet_history = json.load(tweet_history_file)

            tweet_history.append(tweet)

            with open("tweet_history.json", "w", encoding="UTF-8") as tweet_history_file:
                json.dump(tweet_history, tweet_history_file, indent=4)

            return tweet
        else:
            attempts += 1

    # If the function reaches this point, all attempts failed to generate a tweet shorter than 280 characters.
    raise ValueError("Unable to create a tweet within the character limit after 10 attempts.")
