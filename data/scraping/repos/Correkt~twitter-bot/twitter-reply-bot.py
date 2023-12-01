import tweepy
from datetime import datetime, timedelta
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import schedule
import time
import os
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import requests
# Create a new client and connect to the server
# MONGO_URI = os.getenv("MONGO_URI", "YourKey")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_CLIENT = MongoClient(MONGO_URI, server_api=ServerApi('1'))
MONGO_DB = MONGO_CLIENT['correkt']
bot_collection = MONGO_DB['bot']

# Helpful when testing locally
from dotenv import load_dotenv
load_dotenv()

# Load your Twitter and Airtable API keys (preferably from environment variables, config file, or within the railyway app)
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "YourKey")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "YourKey")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "YourKey")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "YourKey")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YourKey")

API_KEY = os.getenv("CORREKT_API_KEY")

# TwitterBot class to help us organize our code and manage shared state
class TwitterBot:
    def __init__(self):
        self.twitter_api = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN,
                                         consumer_key=TWITTER_API_KEY,
                                         consumer_secret=TWITTER_API_SECRET,
                                         access_token=TWITTER_ACCESS_TOKEN,
                                         access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
                                         wait_on_rate_limit=True)
        self.twitter_me_id = self.get_me_id()
        self.tweet_response_limit = 35 # How many tweets to respond to each time the program wakes up

        # Initialize the language model w/ temperature of .5 to induce some creativity
        # self.llm = ChatOpenAI(temperature=.5, openai_api_key=OPENAI_API_KEY, model_name='gpt-4')

        # For statics tracking for each run. This is not persisted anywhere, just logging
        self.mentions_found = 0
        self.mentions_replied = 0
        self.mentions_replied_errors = 0

    # Generate a response using the language model using the template we reviewed in the jupyter notebook (see README)
    def getImages(mentioned_conversation_tweet):
        if ('entities' in mentioned_conversation_tweet and 'media' in mentioned_conversation_tweet):
            media_entities = mentioned_conversation_tweet['entities']['media']
            images = []
            for media_entity in media_entities:
                if (media_entity['type'] == 'photo'):
                    link = media_entity['url']
                    images.append(link)
        return images

    def generate_response(self, mentioned_conversation_tweet):
        # images = getImages(mentioned_conversation_tweet)

        # if images:
        #     data = {"srcs": images, "api_key": API_KEY}
        #     response = requests.post(url="https://correkt.ai/image", json=data)
        #     if response.status_code == 200:
        #         paste = response.json()
        #         comment = ""
        #         for i, src in enumerate(images):
        #             data = paste[src]
        #             if data["ai_probability"] >= 50:
        #                 comment += str(i+1) + ") There's a " + str(data["ai_probability"]) + "% chance this is AI generated." +  "\n"
        #             elif data["ai_probability"] < 50:
        #                 comment += str(i+1) + ") There is a low chance this is AI generated." + "\n"
        #             else:
        #                 comment += str(i+1) + data["error"] + "\n"
        #         comment += "We are currently in alpha testing. Join our Discord server to learn more! [https://discord.gg/Hj9ffKm9ny]"
        #     elif response.status_code == 400:
        #         time.sleep(20)
        #         data = {"srcs": images, "api_key": API_KEY}
        #         response = requests.post(url="https://correkt.ai/image", json=data)
        #         if response.status_code == 200:
        #             paste = response.json()
        #             comment = ""
        #             for i, src in enumerate(images):
        #                 data = paste[src]
        #                 if data["ai_probability"] >= 50:
        #                     comment += str(i+1) + ") There's a " + str(data["ai_probability"]) + "% chance this is AI generated." +  "\n"
        #                 elif data["ai_probability"] < 50:
        #                     comment += str(i+1) + ") There is a low chance this is AI generated." + "\n"
        #                 else:
        #                     comment += str(i+1) + data["error"] + "\n"
        #             comment += "\nWe are currently in alpha testing. Join our Discord server to learn more! [https://discord.gg/Hj9ffKm9ny]"
        #         else:
        #             print(response.status_code)
        #     else:
        #         print(response.status_code)
        #     time.sleep(1)

        info = {"api_key": API_KEY, "sentences": [mentioned_conversation_tweet], "parentText": None}
        response = requests.post(url="https://correkt.ai/validfactcheck", json=info)
        data = response.json()
        if (response.status_code == 200):
            print(1)
            comment = ""
            if (data['checkable'] == True):
                response = requests.post(url="https://correkt.ai/data", json=info)
                print(response)
                data = response.json()
                if (response.status_code == 200):
                    print(2)
                    if (data['checkable'] == False):
                        comment += "The tweet provided could not be fact checked."
                    # could not extract any claims to be fact checked
                    elif (data['results'] == "unfounded" or data['results'] == "untrue"):
                        comment += data["explanation"] + "\n" + data["urls"][0] + data["urls"][1]
                        print(data["explanation"])
                # do something with data["urls"] and data["explanation"]
                    else:
                # text does not contain misinformation
                        comment += "The tweet above does not contain misinformation."
                else:
                # api call failed, sleep 20 seconds and try this whole process again
                    time.sleep(20)
            else:
        # text is not fact checkable
                comment += "This tweet is not fact checkable."
        else:
            time.sleep(20)

        comment += "\n We are currently in alpha testing. Join us on discord! https://discord.gg/Hj9ffKm9ny"
        return comment
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
        
        bot_collection.insert_one({'mentioned_conversation_tweet_id': str(mentioned_conversation_tweet.id)})

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

    # Get mentioned to the user thats authenticated and running the bot.
    # Using a lookback window of 2 hours to avoid parsing over too many tweets
    def get_mentions(self):
        # If doing this in prod make sure to deal with pagination. There could be a lot of mentions!
        # Get current time in UTC
        now = datetime.utcnow()

        # Subtract 2 hours to get the start time
        start_time = now - timedelta(minutes=20)

        # Convert to required string format
        start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        return self.twitter_api.get_users_mentions(id=self.twitter_me_id,
                                                   start_time=start_time_str,
                                                   expansions=['referenced_tweets.id'],
                                                   tweet_fields=['created_at', 'conversation_id']).data
    
    # Checking to see if we've already responded to a mention with what's logged in airtable
    def check_already_responded(self, mentioned_conversation_tweet_id):
        return bot_collection.find_one({'mentioned_conversation_tweet_id': str(mentioned_conversation_tweet_id)}) is not None


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

# The job that we'll schedule to run every X minutes
def job():
    print(f"Job executed at {datetime.utcnow().isoformat()}")
    bot = TwitterBot()
    bot.execute_replies()

if __name__ == "__main__":
    # Schedule the job to run every 5 minutes. Edit to your liking, but watch out for rate limits
    schedule.every(1).minutes.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)