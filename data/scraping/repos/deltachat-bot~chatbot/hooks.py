"""Event Hooks"""
# pylama:ignore=W0603
import asyncio
import json
import logging
import os
from argparse import Namespace
from typing import List, Tuple

import openai
import tiktoken
from deltabot_cli import AttrDict, Bot, BotCli, EventType, const, events

from .openai import get_reply, init_openai
from .orm import init as init_db
from .quota import QuotaManager
from .utils import get_log_level, human_time_duration, run_in_background

cli = BotCli("chatbot", get_log_level())
cfg: dict = {}
quota_manager = QuotaManager(cli, {})
fail_count = 5  # pylint:disable=C0103


@cli.on_init
async def on_init(bot: Bot, _args: Namespace) -> None:
    if not await bot.account.get_config("displayname"):
        await bot.account.set_config("displayname", "ChatBot")
        status = "I am a conversational Delta Chat bot, you can chat with me in private"
        await bot.account.set_config("selfstatus", status)


@cli.on_start
async def _on_start(bot: Bot, args: Namespace) -> None:
    global quota_manager  # pylint:disable=C0103
    path = os.path.join(args.config_dir, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as config:
            cfg.update(json.load(config))
    cfg["openai"] = {"model": "gpt-3.5-turbo", "n": 1, **(cfg.get("openai") or {})}
    api_key = cfg.get("api_key", "")
    assert api_key, "API key is not set"
    await init_openai(api_key, cfg["openai"])

    path = os.path.join(args.config_dir, "sqlite.db")
    await init_db(f"sqlite+aiosqlite:///{path}")

    quota_manager = QuotaManager(cli, cfg)
    run_in_background(quota_manager.cooldown_loop())
    logging.info(
        "Listening for messages at: %s", await bot.account.get_config("configured_addr")
    )


@cli.on(events.RawEvent)
async def log_event(event: AttrDict) -> None:
    if event.type == EventType.INFO:
        logging.info(event.msg)
    elif event.type == EventType.WARNING:
        logging.warning(event.msg)
    elif event.type == EventType.ERROR:
        logging.error(event.msg)


@cli.on(events.MemberListChanged(added=True))
async def _member_added(event: AttrDict) -> None:
    msg = event.message_snapshot
    account = msg.message.account
    if event.member == await account.get_config("configured_addr"):
        await msg.chat.send_text("👋")


@cli.on(events.NewMessage(is_info=False, func=cli.is_not_known_command))
async def _filter_messages(event: AttrDict) -> None:
    global fail_count  # pylint:disable=C0103
    msg = event.message_snapshot
    chat = await msg.chat.get_basic_snapshot()
    if not msg.text or not await _should_reply(msg, chat):
        return

    messages, prompt_tokens = await _get_messages(msg)
    if not messages:
        await msg.chat.send_message(text="TL;DR", quoted_msg=msg.id)
    else:
        global_quota_exceeded = await quota_manager.global_quota_exceeded()
        if global_quota_exceeded:
            cooldown = human_time_duration(await quota_manager.get_global_cooldown())
            await msg.chat.send_message(
                text=f"Quota exceeded, wait for: ⏰ {cooldown}", quoted_msg=msg.id
            )
            return

        quota_exceeded = await quota_manager.quota_exceeded(msg.from_id)
        if quota_exceeded > 0:
            cooldown = human_time_duration(quota_exceeded)
            await msg.chat.send_message(
                text=f"Quota exceeded, wait for: ⏰ {cooldown}", quoted_msg=msg.id
            )
            return

        if quota_manager.is_rate_limited():
            await msg.chat.send_message(
                text="⏰ I'm not available right now, try again later", quoted_msg=msg.id
            )
            return

        try:
            max_tokens = int(cfg["openai"].get("max_tokens") or 0)
            reply = await get_reply(
                str(msg.from_id), messages, max_tokens - prompt_tokens
            )
            logging.debug("bot reply: %s", reply)
            await quota_manager.increase_usage(msg.from_id, reply.usage.total_tokens)
            text = reply.choices[0].message.content.strip()
            await msg.chat.send_message(text=text, quoted_msg=msg.id)
            fail_count = 2
            await asyncio.sleep(1)  # avoid rate limits
        except openai.error.RateLimitError as ex:
            logging.exception(ex)
            await msg.chat.send_message(
                text="⏰ I'm not available right now, try again later", quoted_msg=msg.id
            )
            fail_count = min(fail_count + 1, 60)
            quota_manager.set_rate_limit(60 * fail_count)


async def _get_messages(msg: AttrDict) -> Tuple[List[dict], int]:
    text = ""
    if msg.quote and msg.quote.text:
        text = "> " + msg.quote.text.replace("\n", "\n> ") + "\n\n"

    max_tokens = int(cfg["openai"].get("max_tokens") or 0)
    prompt_tokens = 0
    if max_tokens:
        enc = tiktoken.encoding_for_model(cfg["openai"].get("model"))
        for text2 in (text + msg.text, msg.text):
            tokens = len(enc.encode(text2))
            if tokens <= max_tokens // 2:
                prompt_tokens = tokens
                text = text2
                break
        else:
            text = ""

    return [{"role": "user", "content": text}] if text else [], prompt_tokens


async def _should_reply(msg: AttrDict, chat: AttrDict) -> bool:
    # 1:1 direct chat
    if chat.chat_type == const.ChatType.SINGLE:
        return True

    # mentions
    account = msg.message.account
    selfaddr = await account.get_config("configured_addr")
    displayname = await account.get_config("displayname")
    mention = displayname and msg.text.startswith(f"@{displayname}")
    if msg.text.startswith(selfaddr) or mention:
        return True

    # quote-reply
    if msg.quote and msg.quote.get("message_id"):
        quote = account.get_message_by_id(msg.quote.message_id)
        snapshot = await quote.get_snapshot()
        if snapshot.sender == account.self_contact:
            return True

    return False
