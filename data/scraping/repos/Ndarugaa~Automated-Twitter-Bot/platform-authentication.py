import tweepy
import openai
import telegram
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------TWITTER AUTHENTICATION ----------------------------#

def authenticate_twitter():
    try:
        consumer_key = str(os.getenv('CONSUMER_KEY'))
        consumer_secret = str(os.getenv('API_SECRET_KEY'))
        access_token = str(os.getenv('ACCESS_TOKEN'))
        access_token_secret = str(os.getenv('ACCESS_TOKEN_SECRET'))

        auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)

        # Check if Twitter OAuth1UserHandler authentication is successful
        user = api.verify_credentials()
        print(f"Twitter authentication successful. User: {user.screen_name}")
        
        # V2 Twitter API Authentication
        client = tweepy.Client(
            bearer_token,
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
            wait_on_rate_limit=True,
        )
        return api, client

    except tweepy.TweepyException as e:
        print(f"Twitter authentication failed: {e}")
        return None

# -------------------------- AUTHENTICATE OPENAI -----------------------------------#

def authenticate_openai():
    openai.api_key = str(os.getenv("OPENAI_API_KEY"))

    try:
        response = openai.Completion.create(engine="text-davinci-002", prompt="Testing authentication.")
        print("OpenAI authentication successful.")
        return True

    except Exception as e:
        print(f"OpenAI authentication failed: {e}")
        return False

# -------------------------- AUTHENTICATE TELEGRAM ----------------------------------#

def authenticate_telegram():
    """Authenticates with Telegram and returns the bot token."""

    bot_token = str(os.getenv('TELEGRAM_TOKEN'))

    try:
        bot = telegram.Bot(bot_token)
        return bot.get_me().id
    except telegram.Unauthorized:
        print("Invalid telegram bot token.")
        return None

    bot_id = authenticate_telegram()
    if bot_id:
        print("Bot authenticated successfully.")
    else:
        print("Failed to authenticate bot.")
