from __future__ import annotations
from typing import *

import logging

logger = logging.getLogger(__name__)


from fastapi import FastAPI, Request, Response
import uvicorn

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import asyncio

from slack_bolt.async_app import AsyncApp
from slack_bolt.oauth.async_oauth_settings import AsyncOAuthSettings
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_slack_response import AsyncSlackResponse

import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncEngine

from cogniq.config import (
    APP_ENV,
    APP_URL,
    DATABASE_URL,
    HOST,
    PORT,
    LOG_LEVEL,
    MUTED_LOG_LEVEL,
    SLACK_CLIENT_ID,
    SLACK_CLIENT_SECRET,
    SLACK_SIGNING_SECRET,
)

from .history.openai_history import OpenAIHistory
from .history.anthropic_history import AnthropicHistory
from .search import Search
from .state_store import StateStore
from .installation_store import InstallationStore
from .errors import BotTokenNoneError, BotTokenRevokedError, RefreshTokenInvalidError


class CogniqSlack:
    def __init__(self):
        """
        Slack bot with given configuration, app, and logger.
        """

        self.database_url = DATABASE_URL
        self.engine: AsyncEngine = sqlalchemy.ext.asyncio.create_async_engine(DATABASE_URL)

        self.installation_store = InstallationStore(
            client_id=SLACK_CLIENT_ID,
            client_secret=SLACK_CLIENT_SECRET,
            engine=self.engine,
            install_path=f"{APP_URL}/slack/install",
        )
        self.state_store = StateStore(
            expiration_seconds=120,
            engine=self.engine,
        )
        oauth_settings = AsyncOAuthSettings(
            client_id=SLACK_CLIENT_ID,
            client_secret=SLACK_CLIENT_SECRET,
            scopes=[
                "app_mentions:read",
                "channels:history",
                "chat:write",
                "groups:history",
                "im:history",
                "mpim:history",
            ],
            user_scopes=["search:read"],
            installation_store=self.installation_store,
            user_token_resolution="actor",
            state_store=self.state_store,
            logger=logger,
        )

        app_logger = logging.getLogger(f"{__name__}.slack_bolt")
        app_logger.setLevel(MUTED_LOG_LEVEL)
        self.app = AsyncApp(
            logger=app_logger,
            signing_secret=SLACK_SIGNING_SECRET,
            installation_store=self.installation_store,
            oauth_settings=oauth_settings,
        )

        # Per https://github.com/slackapi/bolt-python/releases/tag/v1.5.0
        self.app.enable_token_revocation_listeners()

        self.app_handler = AsyncSlackRequestHandler(self.app)
        self.api = FastAPI()

        self.anthropic_history = AnthropicHistory(app=self.app)
        self.openai_history = OpenAIHistory(app=self.app)

        # Set defaults
        self.search = Search(cslack=self)

    async def async_setup(self) -> None:
        async with self.engine.begin() as conn:

            def get_tables(sync_conn):
                inspector = sqlalchemy.inspect(sync_conn)
                return inspector.get_table_names()

            table_names = await conn.run_sync(get_tables)
            for table in ["slack_installations", "slack_bots", "slack_oauth_states"]:
                if table not in table_names:
                    raise Exception(f"Table {table} not found in database. Please run migrations with `.venv/bin/alembic upgrade head`.")

    async def start(self):
        """
        This method starts the app.

        It performs the following steps:
        1. Initializes an instance of `AsyncApp` with the Slack bot token, signing secret, and logger.
        2. Creates a `History` object for tracking app events and logging history.
        3. Sets up event registrations for the Slack app by calling the registered functions with the `app` instance.
        4. Logs a message indicating that the Slack app is starting.
        5. If the app environment is set to 'production', the app starts listening on the specified port.
           It awaits the `app.start()` method to start the app server.
        6. If the app environment is set to 'development', the app starts listening on the specified port.
           It will reload the app server if any changes are made to the app code.

        Note:
        - The app will keep running until it is manually stopped or encounters an error.
        """
        logger.info("Starting Slack app!!")
        await self.async_setup()

        @self.api.post("/slack/events")
        async def slack_events(request: Request):
            return await self.app_handler.handle(request)

        @self.api.get("/slack/install")
        async def slack_install(request: Request):
            return await self.app_handler.handle(request)

        @self.api.get("/slack/oauth_redirect")
        async def slack_oauth_redirect(request: Request):
            return await self.app_handler.handle(request)

        @self.api.get("/healthz")
        async def healthz(request: Request):
            return "OK"

        reload = APP_ENV == "development"
        # Run the FastAPI app using Uvicorn
        uvicorn_config = uvicorn.Config(
            self.api,
            host=HOST,
            port=int(PORT),
            log_level=LOG_LEVEL,
            reload=reload,
        )
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()

    async def chat_update(
        self,
        *,
        channel: str,
        text: str,
        ts: str,
        context: Dict[str, Any],
        retry_on_rate_limit: bool = True,
        retry_on_revoked_token: bool = True,
    ) -> AsyncSlackResponse:
        """
        Updates the chat message in the given channel and thread with the given text.
        """
        return await self.api_call(
            method="chat_update",
            channel=channel,
            ts=ts,
            context=context,
            text=text,
            retry_on_rate_limit=retry_on_rate_limit,
            retry_on_revoked_token=retry_on_revoked_token,
        )

    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        context: Dict[str, Any],
        retry_on_rate_limit: bool = True,
        retry_on_revoked_token: bool = True,
    ) -> AsyncSlackResponse:
        """
        Adds the chat message to the given channel and thread with the given text.
        """
        return await self.api_call(
            method="chat_postMessage",
            channel=channel,
            text=text,
            thread_ts=thread_ts,
            context=context,
            retry_on_rate_limit=retry_on_rate_limit,
            retry_on_revoked_token=retry_on_revoked_token,
        )

    def _validate_api_call_params(self, *, bot_token, thread_ts, ts, context) -> None:
        if bot_token is None:
            logger.debug("bot_token is not set. Context: %s", context)
            raise BotTokenNoneError(context=context)

        if thread_ts is not None and not isinstance(thread_ts, str):
            raise ValueError(f"thread_ts should be a string or None, but was {type(thread_ts)}: {thread_ts}")
        if ts is not None and not isinstance(ts, str):
            raise ValueError(f"ts should be a string or None, but was {type(ts)}: {ts}")

    async def api_call(
        self,
        *,
        method: str,
        channel: str,
        text: str,
        thread_ts: str | None = None,
        ts: str | None = None,
        context: Dict[str, Any],
        retry_on_rate_limit: bool = True,
        retry_on_revoked_token: bool = True,
    ) -> AsyncSlackResponse:
        bot_token = context.get("bot_token")
        self._validate_api_call_params(bot_token=bot_token, thread_ts=thread_ts, ts=ts, context=context)

        try:
            logger.debug(f"Calling {method} at {ts} in thread {thread_ts} in channel {channel}")
            return await getattr(self.app.client, method)(
                channel=channel,
                text=text,
                thread_ts=thread_ts,
                ts=ts,
                token=bot_token,
            )
        except SlackApiError as e:
            if e.response["error"] == "ratelimited":
                if retry_on_rate_limit:
                    # Extract the retry value from the headers
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    # Wait for the requested amount of time before retrying
                    await asyncio.sleep(retry_after)
                    return await self.api_call(
                        method=method,
                        channel=channel,
                        text=text,
                        thread_ts=thread_ts,
                        ts=ts,
                        context=context,
                        retry_on_rate_limit=retry_on_rate_limit,
                    )
                else:
                    # Log the rate limit error and move on
                    logger.error("Rate limit hit, not retrying: %s", e)
            if e.response["error"] == "invalid_refresh_token":
                logger.error("Invalid refresh token, not retrying: %s", e)
                raise RefreshTokenInvalidError(message="Invalid refresh token", context=context)
            if e.response["error"] in ["token_revoked", "invalid_auth"]:
                if retry_on_revoked_token:
                    logger.warning("I must have tried to use a revoked or invalid token. I'll try to fetch a newer one.")
                    bot_token = await self.installation_store.async_find_bot_token(context=context)
                    new_context = context.copy()
                    new_context["bot_token"] = bot_token
                    return await self.api_call(
                        method=method,
                        channel=channel,
                        text=text,
                        thread_ts=thread_ts,
                        ts=ts,
                        context=new_context,
                        retry_on_rate_limit=retry_on_rate_limit,
                        retry_on_revoked_token=False,  # Try once, but don't retry again
                    )
                else:
                    raise BotTokenRevokedError(message=str(e), context=context)
            else:
                raise e
