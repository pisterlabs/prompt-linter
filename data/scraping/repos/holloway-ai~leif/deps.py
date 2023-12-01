from app.core.config import settings  # pylint: disable=C0415

from redis import Redis
import cohere
from functools import cached_property

class Database:
    @cached_property
    def connection(self):
        # Initialize the connection to Redis
        # The parameters here are placeholders, replace with your actual parameters
        return Redis( host = settings.REDIS_HOST,
                    port = settings.REDIS_PORT,
                    password = settings.REDIS_PASSWORD)
    @cached_property
    def embedding(self):
        # Initialize the connection to Cohere
        return cohere.Client(settings.COHERE_API_KEY)
      
db = Database()
