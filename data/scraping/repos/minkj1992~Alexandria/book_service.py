import json
from typing import List, Optional

import redis.asyncio as redis
from aredis_om.model import NotFoundError
from langchain.vectorstores.base import VectorStore
from loguru import logger

from src.app.exceptions import BookNotFoundException
from src.infra.langchain import vectorstore
from src.infra.langchain.loader import get_docs_from_urls
from src.infra.redis.connection import get_redis_connection_from_url
from src.infra.redis.models import Book, get_redis_schema_key


async def create_a_book(name: str, description: str) -> str:
    book = await Book(name=name, description=description).save()
    return book.pk


async def create_a_book_chain(book_pk: str, urls: List[str], max_depth: int):
    docs = await get_docs_from_urls(urls, max_depth)
    return await vectorstore.from_docs(docs, index_name=book_pk)


async def get_a_book_chain_schema(
    book_pk: str, redis_conn: Optional[redis.Redis] = None
) -> dict:
    if not redis_conn:
        redis_conn = get_redis_connection_from_url()

    schema_key = get_redis_schema_key(book_pk)
    schema = await redis_conn.get(schema_key)
    return json.loads(schema)


async def create_a_book_chain_schema(
    redis_conn: redis.Redis, book_pk: str, schema: dict
):
    schema_key = get_redis_schema_key(book_pk)
    schema = json.dumps(schema)
    await redis_conn.set(schema_key, schema)


async def get_a_book(book_pk: str) -> Book:
    try:
        book = await Book.get(pk=book_pk)
    except NotFoundError:
        raise BookNotFoundException(book_pk=book_pk)
    except Exception as err:
        logger.exception(err)
        return None
    return book


async def get_a_book_vector(book_pk: str) -> VectorStore:
    schema = await get_a_book_chain_schema(book_pk)
    try:
        vs = await vectorstore.get_vectorstore(book_pk, schema)

    except Exception as err:
        logger.exception(err)
        return None
    return vs
