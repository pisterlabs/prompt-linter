import os
import pathlib
import json
from hashlib import md5
import pathlib
import logging
import argparse
import sys
import re

from dotenv import load_dotenv
import tweepy
import openai
import tiktoken


class TweetScraper:
    def __init__(self, api, data_dir, username):
        self.api = api
        self.data_dir = pathlib.Path(data_dir) / 'tweets' / username
        self.username = username

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger('TweetScraper')
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def all_tweets(self):
        user = self.api.get_user(screen_name=self.username)
        user_id = user.id_str

        max_id = None
        all_tweets = []

        while True:
            tweets = self.__download_tweets(user_id, max_id)
            if not tweets:
                break
            all_tweets.extend(tweets)
            max_id = tweets[-1]['id'] - 1

        self.logger.info(f'Downloaded a total of {len(all_tweets)} tweets.')
        return TweetCollection(all_tweets)

    def __get_cache_filename(self, user_id, max_id):
        hash_input = f'{user_id}-{max_id}' if max_id else f'{user_id}-None'
        cache_key = md5(hash_input.encode('utf-8')).hexdigest()
        prefix = f'{max_id}' if max_id else 'start'
        return self.data_dir / f'{prefix}-{cache_key}.json'

    def __save_tweets_to_file(self, tweets, filepath):
        filepath.write_text(json.dumps([tweet._json for tweet in tweets]))

    def __download_tweets(self, user_id, max_id=None):
        cache_file = self.__get_cache_filename(user_id, max_id)
        if cache_file.exists():
            self.logger.debug(f'Loading tweets from cache: {cache_file}')
            return json.loads(cache_file.read_text())
        else:
            self.logger.info(f'Downloading tweets (max_id: {max_id})')
            tweets = tweepy.Cursor(self.api.user_timeline, user_id=user_id,
                                   max_id=max_id, tweet_mode='extended').items(200)
            tweets = list(tweets)
            self.__save_tweets_to_file(tweets, cache_file)
            return [tweet._json for tweet in tweets]


class TweetCollection:
    def __init__(self, tweets):
        self.tweets = tweets

    def primary_tweet_texts(self):
        selected = []
        for tweet in self.tweets:
            is_reply = tweet['in_reply_to_status_id'] is not None
            is_self_reply = tweet['in_reply_to_user_id'] == tweet['user']['id']
            is_retweet = tweet['retweeted']
            if is_retweet:
                continue
            if is_reply and not is_self_reply:
                continue
            selected.append(tweet['full_text'])

        return selected


class CLI:
    PROJECT_ROOT = pathlib.Path(__file__).parent

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--data-dir', type=str, default=str(CLI.PROJECT_ROOT / 'data'))
        self.parser.add_argument('--username', type=str, required=True)
        self.parser.add_argument(
            '--openai-model', type=str, default=GPTTopicModel.DEFAULT_MODEL)

    def parse_args(self):
        args = self.parser.parse_args()
        return {
            "data_dir": pathlib.Path(args.data_dir),
            "username": args.username,
            "openai_model": args.openai_model
        }


class Bootstrap:
    def __init__(self):
        load_dotenv()

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.twitter_consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
        self.twitter_consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
        self.twitter_token = os.getenv("TWITTER_TOKEN")
        self.twitter_secret = os.getenv("TWITTER_SECRET")

    def setup_openai(self):
        openai.api_key = self.openai_api_key

    def twitter_client(self):
        twitter_auth = tweepy.OAuthHandler(
            self.twitter_consumer_key, self.twitter_consumer_secret)
        twitter_auth.set_access_token(self.twitter_token, self.twitter_secret)
        twitter = tweepy.API(twitter_auth, wait_on_rate_limit=True)
        return twitter


class GPTTopicModel:
    DEFAULT_MODEL = 'gpt-3.5-turbo'

    CHAT_PRELUDE = {
        "role": "system",
        "content": "\n".join([
                "The user will provide you a list of tweets.",
                "Each tweet is separated by \"===\".",
                "Provide 5 most common topics (as in topic modeling) from the tweets.",
                "For each topic, then proivde 5 sub-topics paired with a sentiment in 1-2 words.",
                "Each topic should be at most 1-2 words.",
                "For example: one topic output might look like:",
                "Basketball",
                "  - Pickup basketball - enthsiastic",
                "  - NBA - indifferent",
                "  - March Madness - annoyed",
                "  - Shoes - Excited",
                "  - Lebron James - Angry",
                "Print each topic on a new line. Do not prefix with a number or a bullet. Do not say anything else."
        ])
    }

    MAX_TOKENS_PER_MODEL = {
        'gpt-4': 8_192,
        'gpt-3.5-turbo': 4096,
    }

    def __init__(self, tweet_texts, data_dir, model):
        self.tweet_texts = tweet_texts
        self.data_dir = data_dir / "topics" / model
        self.model = model

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def generate_topics(self):
        chunks = self.__chunked_tweets()
        completions = []
        for chunk in chunks:
            completions.append(self.__topics_for_chunk_with_retry(chunk))

        raw_topics = []
        for completion in completions:
            raw_topics.append(completion['choices'][0]['message']['content'])

        return self.__parse_raw_topics("\n\n".join(raw_topics))

    def __parse_raw_topics(self, raw_topics):
        def in_groups(str):
            lines = str.splitlines()
            groups = []
            current_group = []
            for line in lines:
                if re.match(r"^[a-zA-Z]", line):
                    if len(current_group) > 0:
                        groups.append(current_group)
                        current_group = []
                current_group.append(line)

            if len(current_group) > 0:
                groups.append(current_group)

            return groups

        topics = {}
        groups = in_groups(raw_topics)
        for group in groups:
            topic = group[0]
            if len(topic.strip()) == 0:
                continue
            lines = group[1:]
            subtopic_sentiment_pairs = []
            for line in lines:
                line = re.sub(r"^\s*-\s*", "", line)
                parts = re.split(r"\s+-\s+", line)
                if len(parts) != 2:
                    print(f"warning: malformed line from GPT: {parts}")
                    continue
                subtopic, sentiment = parts
                subtopic_sentiment_pairs.append((subtopic, sentiment))

            if topic not in topics:
                topics[topic] = {}

            for (subtopic, sentiment) in subtopic_sentiment_pairs:
                if subtopic not in topics[topic]:
                    topics[topic][subtopic] = []
                topics[topic][subtopic].append(sentiment)

        return topics

    def __topics_for_chunk_with_retry(self, chunk, retry=True):
        try:
            return self.__topics_for_chunk(chunk)
        except openai.error.OpenAIError as e:
            if retry:
                print(f"Got API error {e}, retrying")
                return self.__topics_for_chunk_with_retry(chunk, retry=False)
            else:
                raise e

    def __topics_for_chunk(self, chunk):
        tweet_block = "\n===\n".join(chunk)
        messages = [GPTTopicModel.CHAT_PRELUDE, {
            "role": "user", "content": tweet_block
        }]

        digest = md5(json.dumps(messages).encode('utf-8')).hexdigest()
        cache_file = self.data_dir / f"topics_{digest}.json"

        if cache_file.exists():
            print("Getting completion from cache")
            return json.loads(cache_file.read_text())
        else:
            print("Getting completion from API")
            response = openai.ChatCompletion.create(
                messages=messages,
                model=self.model,
                temperature=0.0,
                max_tokens=100
            )
            raw = response.to_dict_recursive()
            cache_file.write_text(json.dumps(raw))
            return raw

    def __chunked_tweets(self):
        encoding = tiktoken.encoding_for_model(self.model)
        chunks = []
        current_chunk = []
        current_cunk_size = 0

        for tweet in self.tweet_texts:
            size = len(encoding.encode(tweet))
            if current_cunk_size + size > self.__max_chunk_size():
                chunks.append(current_chunk)
                current_chunk = []
                current_cunk_size = 0
            current_chunk.append(tweet)
            current_cunk_size += size

        if len(current_chunk) > 0:
            chunks.append(current_chunk)

        return chunks

    def __max_chunk_size(self):
        ceiling = GPTTopicModel.MAX_TOKENS_PER_MODEL[self.model]
        return ceiling - 1000  # Leave room


def main():
    args = CLI().parse_args()
    bootstrap = Bootstrap()

    bootstrap.setup_openai()
    twitter = bootstrap.twitter_client()

    scraper = TweetScraper(twitter, args['data_dir'], args['username'])
    tweets = scraper.all_tweets().primary_tweet_texts()

    modeler = GPTTopicModel(
        tweets, args['data_dir'], args['openai_model']
    )
    print(json.dumps(modeler.generate_topics(), indent=2))


if __name__ == '__main__':
    main()
