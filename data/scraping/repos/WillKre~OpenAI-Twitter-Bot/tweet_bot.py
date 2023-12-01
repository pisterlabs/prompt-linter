import tweepy
import json
import openai
import boto3
from botocore.exceptions import ClientError

class OpenAIBot:
    def __init__(self, bot_name, chat_gpt_model, prompt_prefix, aws_region_name, aws_secret_name, aws_dynamodb_table_name):
        # AWS Config
        self.region_name = aws_region_name
        self.secret_name = aws_secret_name
        self.table_name = aws_dynamodb_table_name

        # AWS Clients / Creds
        self.credentials = self.get_secrets_from_secrets_manager()
        self.dynamodb_client = boto3.client(service_name='dynamodb', region_name=self.region_name)

        # Bot Config
        self.bot_name = bot_name
        self.chat_gpt_model = chat_gpt_model
        self.prompt_prefix = prompt_prefix

        # Tweepy
        self.tweepy_client = self.get_tweepy_client()

    def get_secrets_from_secrets_manager(self):
        """
        Retrieves secrets from AWS Secrets Manager.

        Return:
            dict: The secrets as a dictionary.

        Raises:
            Exception: If any error occurs during the secret retrieval.
        """
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=self.region_name
        )
        try:
            get_secret_value_response = client.get_secret_value(SecretId=self.secret_name)
            return json.loads(get_secret_value_response['SecretString'])
        except Exception as e:
            print(f'Failed to fetch credentials from AWS secrets manager. Error: {str(e)}')
            raise


    def get_tweepy_client(self):
        """
        Constructs and returns a Tweepy client configured with the provided credentials.

        Return:
            tweepy.Client: A Tweepy client.
        """
        return tweepy.Client(
            bearer_token=self.credentials['bearer_token'],
            consumer_key=self.credentials['consumer_key'],
            consumer_secret=self.credentials['consumer_secret'],
            access_token=self.credentials['access_token'],
            access_token_secret=self.credentials['access_token_secret']
        )

    def get_tweets(self):
        """
        Searches and returns recent tweets mentioning the bot.

        Return:
            tweepy.Response: A Tweepy response object containing the found tweets.
        """
        query = f"@{self.bot_name}"
        max_results = 10

        return self.tweepy_client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["referenced_tweets"])

    def process_tweet(self, tweet):
        """
        Processes the parent tweet using an instruction tuned large language model (LLM).

        Args:
            tweet (tweepy.Tweet): A Tweepy Tweet object representing the tweet to process.

        Return:
            str: Processed text from the tweet's content.
        """
        try:
            openai.api_key = self.credentials['open_ai']
            completion = openai.ChatCompletion.create(
                model=self.chat_gpt_model,
                messages=[
                    {"role": "system", "content": self.prompt_prefix},
                    {"role": "user", "content": tweet.data.text}
                ]
            )
            processed_tweet = completion.choices[0].message.content
            print(f"Processed tweet is: {processed_tweet}")
            return processed_tweet
        except:
            print(f"Something went wrong attempting to process tweet: {tweet.data.id}")
            return None

    def respond_to_tweet(self, tweet_id, processed_tweet):
        """
        Creates a tweet in response to the parent.

        Args:
            tweet_id: The id of the tweet which tagged our bot.
            processed_tweet: The OpenAI processed version of the tweet text.
        """
        self.tweepy_client.create_tweet(
            text=processed_tweet,
            in_reply_to_tweet_id=tweet_id
        )
        print(f"Successfully responded to tweet: {tweet_id}")

    def respond_to_tweets(self, tweets):
        """
        Fetches the "parent tweet" (either the tweet that is quoted or the tweet that is replied to), and replies with a summary.

        Args:
            tweets (tweepy.Response): A Tweepy response object containing the tweets to respond to.
        """
        if tweets.data is not None:
            for tweet in tweets.data:
                if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets is not None:
                    for referenced_tweet in tweet.referenced_tweets:

                        if not self.has_tweet_been_responded_to(tweet.id):
                            parent_tweet = self.tweepy_client.get_tweet(referenced_tweet.id)
                            processed_tweet = self.process_tweet(parent_tweet)
                            if processed_tweet is not None:
                                self.add_tweet_id_to_database(tweet.id)
                                self.respond_to_tweet(tweet.id, processed_tweet)
                        else:
                            print(f"Tweet {tweet.id} has already been responded to!")
                else:
                    print(f"Unable to retrieve parent tweet for tweet id: {tweet.id}")
        else:
            print("No tweets found!")

    def add_tweet_id_to_database(self, tweet_id):
        """
        Adds the tweet_id to the DynamoDB database so that it's tracked as responded.

        Args:
            tweet_id (number): The tweet id of the tweet which tagged the bot.
        """
        try:
            self.dynamodb_client.put_item(TableName=self.table_name, Item={'tweet_id': { 'S': str(tweet_id) }})
        except ClientError as e:
            print(f"Unable to add tweet: {e}")

    def has_tweet_been_responded_to(self, tweet_id):
        """
        Checks if the tweet (the one which tagged, not the parent) has been responded to.

        Args:
            tweet_id (number): The tweet id of the tweet which tagged the bot.

        Return:
            True if the tweet_id is in the database, False if not.
        """
        response = self.dynamodb_client.get_item(TableName=self.table_name, Key={'tweet_id': { 'S': str(tweet_id)}})

        if 'Item' in response:
            return True
        else:
            return False

