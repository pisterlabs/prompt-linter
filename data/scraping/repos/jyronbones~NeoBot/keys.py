import os
import openai
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_TOKEN")

MODEL_ENGINE = os.getenv("OPENAI_MODEL_ENGINE")
openai.api_key = os.getenv("OPENAI_API_KEY")

NEWS_API_KEY = os.getenv("NEWSAPI_KEY")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

MUSICMATCH_API_KEY = os.getenv("MUSIC_MATCH_API_KEY")

SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

YOUR_WEATHERSTACK_API_KEY = os.getenv("YOUR_WEATHERSTACK_API_KEY")

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

DISCORD_LOGS_DB = os.getenv("DISCORD_LOGS_DB")

DISCORD_LOGS_TABLE_NAME = os.getenv("DISCORD_LOGS_TABLE_NAME")

DB_SERVER_NAME = os.getenv("DB_SERVER_NAME")

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
