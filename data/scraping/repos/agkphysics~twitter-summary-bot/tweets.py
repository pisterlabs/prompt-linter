import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import openai
import tweepy
from aws_lambda_powertools.logging import Logger

from keys import (
    APP_BEARER_TOKEN,
    CONSUMER_KEY,
    CONSUMER_SECRET,
    OAUTH1_BOT_ACCESS_TOKEN,
    OAUTH1_BOT_TOKEN_SECRET,
    OPENAI_API_KEY,
)
from utils import build_tweet_tree, enumerate_tweet_tree, get_parent

openai.api_key = OPENAI_API_KEY
BOT_USER_ID = int(os.environ["BOT_USER_ID"])

logger = Logger(service="twitter-webhook", child=True)

tw_client = tweepy.Client(
    bearer_token=APP_BEARER_TOKEN,
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=OAUTH1_BOT_ACCESS_TOKEN,
    access_token_secret=OAUTH1_BOT_TOKEN_SECRET,
)


class TweetTooOldError(Exception):
    """Exception raised when a tweet is too old to be replied to."""


class InvalidTaggingTweetError(Exception):
    """Exception raised when a tweet is invalid."""


class TweetNotFoundError(Exception):
    """Exception raised when a tweet is not found."""


class CannotAccessTweetError(Exception):
    """Exception raised when a tweet is not accessible."""


def handle_errors(errors: list[dict[str, Any]]):
    error = errors[0]
    if error["type"] == "https://api.twitter.com/2/problems/resource-not-found":
        raise TweetNotFoundError(error["detail"])
    if (
        error["type"]
        == "https://api.twitter.com/2/problems/not-authorized-for-resource"
    ):
        raise CannotAccessTweetError(error["detail"])
    raise RuntimeError(f"Error from Twitter API: {error}")


def get_conversation_id(tweet_id: int) -> int:
    """Get the conversation ID of the tagging tweet.

    Args
    ----
    tweet_id: int
        The ID of the tagging tweet.

    Returns
    -------
    Optional[int]
        The conversation ID of the tweet, or None if the tweet doesn't exist.
    """

    resp = tw_client.get_tweet(
        tweet_id,
        tweet_fields=["conversation_id", "referenced_tweets"],
        expansions=["referenced_tweets.id"],
    )
    tweet: tweepy.Tweet = resp.data
    if resp.errors:
        handle_errors(resp.errors)
    logger.debug("Tagging tweet: %s", tweet)

    if tweet.referenced_tweets:
        for ref in tweet.referenced_tweets:
            if ref.type == "quoted":  # Tagging tweet is a quote tweet
                qt: tweepy.Tweet
                for qt in resp.includes["tweets"]:
                    if qt.id == ref.id:
                        return qt.conversation_id
    return tweet.conversation_id


def get_thread_tweets(conv_id: int, author_id: int) -> tweepy.Response:
    """Get the tweets in a thread.

    Args
    ----
    conv_id: int
        The ID of the conversation to get the tweets of.
    author_id: int
        The ID of the author of the conversation.

    Returns
    -------
    tweepy.Response
        The response from the Twitter API.
    """

    end_time = datetime.utcnow() - timedelta(seconds=10)
    end_time_str = end_time.isoformat("T", timespec="seconds") + "Z"

    def _get_tweets() -> tweepy.Response:
        return tw_client.search_recent_tweets(
            f"from:{author_id} to:{author_id} conversation_id:{conv_id}",
            max_results=100,
            tweet_fields=["referenced_tweets", "conversation_id", "author_id"],
            expansions=["referenced_tweets.id"],
            end_time=end_time_str,
        )

    while (resp := _get_tweets()).meta["result_count"] == 0:
        time.sleep(3)  # Wait for API to catch up
    return resp


def get_gpt_summary(thread: list[str]) -> str:
    """Get a summary of a thread using GPT-3.

    Args
    ----
    thread: str
        The thread to summarize.

    Returns
    -------
    str
        The summary of the thread.
    """
    prompt = "<tweet>" + "</tweet><tweet>".join(thread) + "</tweet>"
    prompt = (
        f"{prompt}\nSummarize the above into a single 280 character Tweet:\n<tweet>"
    )
    summary = (
        openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=70,
            stop="</tweet>",
        )
        .choices[0]
        .text.strip()
    )
    return summary


def limit_summary(summary: str) -> str:
    """Limit the summary to 280 characters.

    Args
    ----
    summary: str
        The summary to limit.

    Returns
    -------
    str
        The summary, constrained to 280 characters.
    """
    if len(summary) <= 280:
        return summary

    removed = 0
    words = summary.split()
    for i in range(len(words) - 1, 0, -1):
        removed += len(words[i]) + 1
        if len(summary) - removed <= 280:
            summary = " ".join(words[:i])[:280]
            break
    return summary


class TweetHandler:
    def __init__(self, tweet: dict[str, Any]) -> None:
        self.tweet = tweet
        self.conv_id = get_conversation_id(tweet["id"])

    def handle(self) -> bool:
        try:
            thread = self.get_tweet_thread()
            logger.info(f"Thread:\n\n{thread}")
            summary = get_gpt_summary(thread)
            logger.info(f"GPT summary:\n\n{summary}")
            summary = limit_summary(summary)
        except TweetTooOldError as e:
            logger.info(e)
            self.reply_to_tweet(
                "I'm sorry but I can't analyse threads older than 7 days"
            )
            return False
        except TweetNotFoundError as e:
            logger.info(e)
            self.reply_to_tweet("I'm sorry but I can't find that tweet")
            return False
        except CannotAccessTweetError as e:
            logger.info(e)
            self.reply_to_tweet("I'm sorry but I can't access that tweet")
            return False
        except InvalidTaggingTweetError as e:
            logger.info(e)
            # Cannot reply to tweet here because this will always get
            # triggered by replies below the bot
            return False
        except Exception as e:
            logger.exception(f"Error getting tweet thread: {e}")
            return False

        self.reply_to_tweet(summary)
        return True

    def reply_to_tweet(self, summary: str) -> None:
        logger.info(f"Replying to tweet {self.tweet['id']} with {summary}")
        if os.environ.get("DEBUG", "0") == "1":
            return
        tw_client.create_tweet(
            text=summary,
            in_reply_to_tweet_id=self.tweet["id"],
            exclude_reply_user_ids=[BOT_USER_ID, self.thread_author],
        )

    def get_tweet_thread(self) -> list[str]:
        resp = tw_client.get_tweet(self.conv_id, tweet_fields=["author_id,created_at"])
        if resp.errors:
            handle_errors(resp.errors)

        tweet: tweepy.Tweet
        tweet = resp.data
        logger.debug("Thread start tweet: %s", tweet)

        self.thread_author = tweet.author_id
        text = tweet.text

        if self.thread_author != self.tweet["in_reply_to_user_id"]:
            raise InvalidTaggingTweetError("Tweet is not a reply to the thread author")
        if tweet.created_at < datetime.now(timezone.utc) - timedelta(days=7):
            raise TweetTooOldError("Tweet is older than 7 days")

        data = get_thread_tweets(self.conv_id, self.thread_author)
        logger.debug("Tweet thread data: %s", data.data)

        # Need to get the included tweets, since sometimes the API doesn't return all
        # the tweets
        tweets: dict[int, str] = {self.conv_id: text}
        parents: dict[int, int] = {}
        for tweet in data.includes["tweets"] + data.data:
            if (
                tweet.author_id == self.thread_author
                and tweet.conversation_id == self.conv_id
            ):
                tweets[tweet.id] = tweet.text
            if (p := get_parent(tweet)) is not None:
                parents[tweet.id] = p

        tree = build_tweet_tree(parents)
        logger.debug("Tweet tree: %s", tree)

        # We need to ignore the tagging tweet, in case we're tagged by the thread
        # author.
        conversation = [
            tweets[x]
            for x in enumerate_tweet_tree(tree, self.conv_id)
            if x != self.tweet["id"]
        ]
        return conversation
