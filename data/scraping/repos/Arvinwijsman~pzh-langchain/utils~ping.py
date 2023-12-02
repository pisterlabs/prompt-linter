import openai
from utils.vectorstores import _init_redis_connection
from utils.settings import settings


def ping_redis(db=0):
    print(f"Pinging {settings.redis_dsn}")
    try:
        conn = _init_redis_connection(db)
        conn.ping()
    except ConnectionError as e:
        print(e)
        return False

    return True


def ping_openai():
    openai.api_key = settings.openai_api_key
    try:
        print("Pinging openAI to verify key.")
        openai.Model.list()
    except Exception as e:
        print(f"OpenAI connection failed with error: {e}")
        return False

    return True
