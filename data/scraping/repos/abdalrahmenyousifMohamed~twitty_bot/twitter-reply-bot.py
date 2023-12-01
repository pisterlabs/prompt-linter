import tweepy
from airtable import Airtable
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import schedule
import time
import os

# Helpful when testing locally
from dotenv import load_dotenv
load_dotenv()

# Load your Twitter and Airtable API keys (preferably from environment variables, config file, or within the railyway app)
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "YourKey")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "YourKey")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "YourKey")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "YourKey")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YourKey")
# Check for all new mentions and it already responded
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY", "YourKey")
AIRTABLE_BASE_KEY = os.getenv("AIRTABLE_BASE_KEY", "YourKey")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "YourKey")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YourKey")

# TwitterBot class to help us organize our code and manage shared state
class TwitterBot:
    def __init__(self):
        self.twitter_api = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN,
                                         consumer_key=TWITTER_API_KEY,
                                         consumer_secret=TWITTER_API_SECRET,
                                         access_token=TWITTER_ACCESS_TOKEN,
                                         access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
                                         wait_on_rate_limit=True)
        self.auth = tweepy.OAuth1UserHandler(TWITTER_API_KEY,TWITTER_API_SECRET,TWITTER_ACCESS_TOKEN,TWITTER_ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(self.auth)

        self.airtable = Airtable(AIRTABLE_BASE_KEY, AIRTABLE_TABLE_NAME, AIRTABLE_API_KEY)
        self.twitter_me_id = self.get_me_id()
        self.tweet_response_limit = 35 # How many tweets to respond to each time the program wakes up

        # Initialize the language model w/ temperature of .5 to induce some creativity
        self.llm = ChatOpenAI(temperature=.5, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')

        # For statics tracking for each run. This is not persisted anywhere, just logging
        self.mentions_found = 0
        self.mentions_replied = 0
        self.mentions_replied_errors = 0

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
    
        # Generate a response using the language model
    def respond_to_mention(self, mention, mentioned_conversation_tweet):
        response_text = self.generate_response(mentioned_conversation_tweet.text)
        
        # Try and create the response to the tweet. If it fails, log it and move on
        try:
            response_tweet = self.twitter_api.create_tweet(text=response_text, in_reply_to_tweet_id=mention.id)
            self.mentions_replied += 1
        except Exception as e:
            print (e)
            self.mentions_replied_errors += 1
            return
        
        # Log the response in airtable if it was successful
        self.airtable.insert({
            'mentioned_conversation_tweet_id': str(mentioned_conversation_tweet.id),
            'mentioned_conversation_tweet_text': mentioned_conversation_tweet.text,
            'tweet_response_id': response_tweet.data['id'],
            'tweet_response_text': response_text,
            'tweet_response_created_at' : datetime.utcnow().isoformat(),
            'mentioned_at' : mention.created_at.isoformat()
        })
        return True
    
    # Returns the ID of the authenticated user for tweet creation purposes
    def get_me_id(self):
        return self.twitter_api.get_me()[0].id
    
    # Returns the parent tweet text of a mention if it exists. Otherwise returns None
    # We use this to since we want to respond to the parent tweet, not the mention itself
    def get_mention_conversation_tweet(self, mention):
        # Check to see if mention has a field 'conversation_id' and if it's not null
        if mention.conversation_id is not None:
            conversation_tweet = self.twitter_api.get_tweet(mention.conversation_id).data
            return conversation_tweet
        return None


    def get_mentions(self):
        # If doing this in prod make sure to deal with pagination. There could be a lot of mentions!
        # Get current time in UTC
        now = datetime.utcnow()

        """
        Real-time Monitoring: The lookback window allows you to focus on recent activity. 
        By looking back only 20 minutes, you can effectively monitor and retrieve mentions 
        that have occurred within a short time frame. 
        This is particularly useful for real-time or near-real-time applications where you want to stay up-to-date with interactions and mentions on Twitter.
        """
        start_time = now - timedelta(minutes=20)

        # Convert to required string format
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return self.twitter_api.get_users_mentions(id=self.twitter_me_id,
                                                   start_time=start_time_str,
                                                   expansions=['referenced_tweets.id'],
                                                   tweet_fields=['created_at', 'conversation_id']).data

    # Checking to see if we've already responded to a mention with what's logged in airtable
    def check_already_responded(self, mentioned_conversation_tweet_id):
        records = self.airtable.get_all(view='Grid view')
        for record in records:
            if record['fields'].get('mentioned_conversation_tweet_id') == str(mentioned_conversation_tweet_id):
                return True
        return False

    # Run through all mentioned tweets and generate a response
    def respond_to_mentions(self):
        mentions = self.get_mentions()

        # If no mentions, just return
        if not mentions:
            print("No mentions found")
            return
        
        self.mentions_found = len(mentions)

        for mention in mentions[:self.tweet_response_limit]:
            # Getting the mention's conversation tweet
            mentioned_conversation_tweet = self.get_mention_conversation_tweet(mention)
            
            # If the mention *is* the conversation or you've already responded, skip it and don't respond
            if (mentioned_conversation_tweet.id != mention.id
                and not self.check_already_responded(mentioned_conversation_tweet.id)):

                self.respond_to_mention(mention, mentioned_conversation_tweet)
        return True
    
        # The main entry point for the bot with some logging
    def execute_replies(self):
        print (f"Starting Job: {datetime.utcnow().isoformat()}")
        self.respond_to_mentions()
        print (f"Finished Job: {datetime.utcnow().isoformat()}, Found: {self.mentions_found}, Replied: {self.mentions_replied}, Errors: {self.mentions_replied_errors}")

def job():
    print(f"Job executed at {datetime.utcnow().isoformat()}")
    bot = TwitterBot()
    bot.execute_replies()

if __name__ == "__main__":
    # Schedule the job to run every 5 minutes.
    schedule.every(3).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)