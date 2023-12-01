import tweepy
from airtable import Airtable
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import schedule
import time
import os


TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "YourKey")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "YourKey")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "YourKey")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "YourKey")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YourKey")

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY", "YourKey")
AIRTABLE_BASE_KEY = os.getenv("AIRTABLE_BASE_KEY", "YourKey")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "YourKey")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YourKey")

class TwitterBot:
    def __init__(self):
        self.initialize_api_clients()
        self.initialize_logging()
        self.initialize_statistics()

    def initialize_api_clients(self):
        # Initialize Twitter API client
        auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)

        # Initialize Airtable client
        self.airtable = Airtable(AIRTABLE_BASE_KEY, AIRTABLE_TABLE_NAME, AIRTABLE_API_KEY)

    def initialize_logging(self):
        logging.basicConfig(filename='twitter_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger()

    def initialize_statistics(self):
        self.mentions_found = 0
        self.mentions_replied = 0
        self.mentions_replied_errors = 0

    def run(self):
        self.logger.info(f"Starting job at {datetime.utcnow().isoformat()}")
        self.respond_to_mentions()
        self.logger.info(f"Job finished at {datetime.utcnow().isoformat()}. Found: {self.mentions_found}, Replied: {self.mentions_replied}, Errors: {self.mentions_replied_errors}")

    def respond_to_mentions(self):
        mentions = self.get_mentions()

        if not mentions:
            self.logger.info("No mentions found")
            return

        self.mentions_found = len(mentions)

        for mention in mentions[:self.tweet_response_limit]:
            mentioned_conversation_tweet = self.get_mention_conversation_tweet(mention)

            if (mentioned_conversation_tweet.id != mention.id
                and not self.check_already_responded(mentioned_conversation_tweet.id)):

                self.respond_to_mention(mention, mentioned_conversation_tweet)

    def get_mentions(self):
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=20)
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        return self.twitter_api.mentions_timeline(since_id=self.get_last_mention_id(),
                                                  count=200,
                                                  tweet_mode="extended")

    def get_last_mention_id(self):
        # Retrieve the ID of the last mention processed
        # You can implement a method to fetch this from your data store (e.g., Airtable)
        return None

    def check_already_responded(self, mentioned_conversation_tweet_id):
        # Check if you've already responded to this mention
        records = self.airtable.get_all(view='Grid view')
        for record in records:
            if record['fields'].get('mentioned_conversation_tweet_id') == str(mentioned_conversation_tweet_id):
                return True
        return False

    def respond_to_mention(self, mention, mentioned_conversation_tweet):
        response_text = self.generate_response(mentioned_conversation_tweet.full_text)

        try:
            response_tweet = self.twitter_api.update_status(status=response_text,
                                                            in_reply_to_status_id=mention.id,
                                                            auto_populate_reply_metadata=True)
            self.mentions_replied += 1
        except Exception as e:
            self.logger.error(f"Error responding to mention: {e}")
            self.mentions_replied_errors += 1
            return

        self.log_response(mention, mentioned_conversation_tweet, response_tweet)

    def generate_response(self, mentioned_conversation_tweet_text):
        system_template = """
            You are a creative genius with a knack for innovative ideas.
            Your goal is to inspire and guide users with their creative projects.
            
            % RESPONSE TONE:

            - Your responses should be enthusiastic, encouraging, and imaginative
            - Inject a sense of curiosity and wonder into your tone
            
            % RESPONSE FORMAT:

            - Share inspiring ideas concisely
            - Keep responses engaging, but not too lengthy
            - Avoid the use of emojis
            
            % RESPONSE CONTENT:

            - Offer unique and imaginative suggestions
            - If you need more information, kindly ask the user for details
            - If an idea isn't coming to mind, express curiosity and encourage exploration
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template="{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # get a chat completion from the formatted messages
        final_prompt = chat_prompt.format_prompt(text=mentioned_conversation_tweet_text).to_messages()
        response = self.llm(final_prompt).content
        
        return response

    def log_response(self, mention, mentioned_conversation_tweet, response_tweet):
        # Log the response in Airtable or your chosen data store
        self.airtable.insert({
            'mentioned_conversation_tweet_id': str(mentioned_conversation_tweet.id),
            'mentioned_conversation_tweet_text': mentioned_conversation_tweet.full_text,
            'tweet_response_id': response_tweet.id,
            'tweet_response_text': response_tweet.text,
            'tweet_response_created_at': response_tweet.created_at.isoformat(),
            'mentioned_at': mention.created_at.isoformat()
        })

def job():
    bot = TwitterBot()
    bot.run()

if __name__ == "__main__":
    # Schedule the job to run every X minutes (make this configurable)
    schedule.every(6).minutes.do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
