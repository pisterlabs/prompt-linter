from redis import Redis, ConnectionError
from sqlalchemy.exc import OperationalError, ArgumentError
import psycopg2
from sqlalchemy import create_engine
from langchain.cache import InMemoryCache, SQLiteCache, SQLAlchemyCache, RedisCache
import langchain
import os


def create_cache(style="in_memory", **kwargs):
    if style == "in_memory":
        return create_in_memory_cache()
    elif style == "sqlite":
        return  create_sqlite_cache(**kwargs)
    elif style == "sqlalchemy":
        return  create_sqlalchemy_cache(**kwargs)
    elif style == "redis":
        return create_redis_cache()
    else:
        raise ValueError("Invalid cache style: {}".format(style))


def create_in_memory_cache():
    _llm_cache = InMemoryCache()
    return _llm_cache


def create_redis_cache():
    redis_client = Redis()
    _llm_cache = None
    try:
        # Check if the local Redis instance is running
        if redis_client.ping():
            print("Local Redis instance is running.")
            _llm_cache = RedisCache(redis_=redis_client)
        else:
            print("Local Redis instance is not running.")
    except ConnectionError:
        print("\n... Unable to connect to the local Redis instance ... [Is the local Redis instance running?] ...\n")
    return _llm_cache


def create_sqlite_cache(db_path=".langchain.db", remove_existing_db=True, **kwargs):
    # !rm.langchain.db
    if remove_existing_db and os.path.isfile(db_path):
        os.remove(db_path)

    _llm_cache = SQLiteCache(database_path=db_path)

    return _llm_cache


def create_sqlalchemy_cache(engine_path, **kwargs):
    _llm_cache = None
    try:
        engine = create_engine(engine_path)
        _llm_cache = SQLAlchemyCache(engine)

        # Test connection by executing a simple query
        with engine.connect() as connection:
            connection.execute("SELECT 1")

    except ArgumentError as arg_error:
        print("Invalid connection string. Check the format and parameters:", arg_error)

    except OperationalError as op_error:
        print("Could not connect to the PostgreSQL server. Possible issues include:", op_error)
        print("1. PostgreSQL service not running.")
        print("2. Incorrect username or password.")
        print("3. Firewall or network issue blocking the connection.")
        print("4. Database does not exist.")

    except psycopg2.errors.DependentObjectsStillExist as do_error:
        print("Unable to drop the database due to dependent objects. Consider using CASCADE:", do_error)

    except Exception as general_error:
        print("An unexpected error occurred:", general_error)

    return _llm_cache


