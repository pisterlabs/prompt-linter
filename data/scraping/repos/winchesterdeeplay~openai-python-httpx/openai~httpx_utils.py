import logging

import httpx

import openai


logging.basicConfig(level=logging.INFO)


def setup_custom_sync_session(**params) -> None:
    """
    Setup custom sync session (httpx.Client) to use for all API calls.

    https://www.python-httpx.org/api/#client
    """
    openai.sync_session = httpx.Client(**params)


def setup_custom_async_session(**params) -> None:
    """
    Setup custom async session (httpx.AsyncClient) to use for all API calls.

    https://www.python-httpx.org/api/#asyncclient
    """
    openai.async_session = httpx.AsyncClient(**params)


async def force_init_pulls() -> None:
    """
    Force initialization of the connection pools for sync and async sessions.

    https://en.wikipedia.org/wiki/HTTP_persistent_connection
    """
    if openai.api_base is None or openai.api_key is None:
        raise ValueError("Please setup api_base and api_key before using this function")
    force_init_sync_pulls()
    await force_init_async_pulls()


def force_init_sync_pulls() -> None:
    """
    Force initialization of the connection pool for sync session.
    """
    if openai.api_base is None or openai.api_key is None:
        raise ValueError("Please setup api_base and api_key before using this function")
    openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test message. Ignore it."}]
    )
    logging.info(f"HTTP pools initialized for Sync session")


async def force_init_async_pulls() -> None:
    """
    Force initialization of the connection pool for async session.
    """
    if openai.api_base is None or openai.api_key is None:
        raise ValueError("Please setup api_base and api_key before using this function")
    await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test message. Ignore it."}]
    )
    logging.info(f"HTTP pools initialized for Sync session")


def reset_sessions() -> None:
    """
    Reset sync and async sessions to default values.
    """
    openai.sync_session = openai.init_session(sync=True)
    openai.async_session = openai.init_session(sync=False)
    logging.info(f"HTTP pools reset to default values")
