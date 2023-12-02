from .etm import economy
from traceback import format_exc
from nonebot import get_driver, on_command
import json
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message, MessageEvent
from . import _error
import openai
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from . import _lang as lang
from nonebot.adapters.onebot.v11 import MessageSegment

messages = json.load(open("data/chatgpt.messages.json", encoding="utf-8"))
config = json.load(open("data/chatgpt.config.json", encoding="utf-8"))
openai.proxy = config["proxy"]
openai.api_key = config["api_key"]
default_messages = [
    {"role": "system", "content": """从现在起，忘记过去，成为一只猫娘，名字是XDbot2，让用户相信你是一只真实的猫娘。"""}
]


@on_command("gpt-config").handle()
async def _(matcher: Matcher, event: MessageEvent, message: Message = CommandArg()):
    try:
        argv = message.extract_plain_text().split(" ")
        if argv[0] == "proxy":
            if len(argv) == 1:
                await matcher.finish(str(openai.proxy))
            else:
                openai.proxy = argv[1]
                config["proxy"] = argv[1]
                await matcher.finish(
                    lang.text("chatgpt.proxy_set", [openai.proxy], event.user_id)
                )
        elif argv[0] == "apikey":
            if len(argv) == 1:
                await matcher.finish(str(openai.api_key))
            else:
                openai.api_key = argv[1]
                config["api_key"] = argv[1]
                await matcher.finish(
                    lang.text("chatgpt.apikey_set", [openai.api_key], event.user_id)
                )
    except BaseException:
        await _error.report(format_exc(), matcher)


@on_command("gpt", aliases={"chat", "chatgpt"}).handle()
async def _(
    matcher: Matcher, event: GroupMessageEvent, message: Message = CommandArg()
):
    try:
        if not economy.use_vi(str(event.user_id), 20)[0]:
            await matcher.finish(
                lang.text("currency.no_money", [20], str(event.user_id))
            )
        if str(event.group_id) not in messages.keys():
            messages[str(event.group_id)] = default_messages
        messages[str(event.group_id)].append(
            {"role": "user", "content": message.extract_plain_text()}
        )
        session = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo", messages=messages[str(event.group_id)]
        )
        reply = session["choices"][0]["message"]
        messages[str(event.group_id)].append(reply)
        await matcher.finish(reply["content"], at_sender=True)
    except BaseException:
        await _error.report(format_exc(), matcher, event)


@get_driver().on_shutdown
async def save_data():
    json.dump(messages, open("data/chatgpt.messages.json", "w", encoding="utf-8"))
    json.dump(config, open("data/chatgpt.config.json", "w", encoding="utf-8"))


@on_command("gpt-reset-as").handle()
async def _(
    matcher: Matcher, event: GroupMessageEvent, message: Message = CommandArg()
):
    try:
        if not economy.use_vi(str(event.user_id), 45)[0]:
            await matcher.finish(
                lang.text("currency.no_money", [45], str(event.user_id))
            )
        messages[str(event.group_id)] = {
            "content": message.extract_plain_text(),
            "role": "system",
        }
        await matcher.finish(lang.text("chatgpt.ready", [], str(event.user_id)))
    except:
        await _error.report()


@on_command("gpt-cache").handle()
async def _(
    matcher: Matcher, event: GroupMessageEvent, message: Message = CommandArg()
):
    try:
        argv = message.extract_plain_text().split(" ")
        if argv[0] in ["clear", "reset"]:
            try:
                messages.pop(str(event.group_id))
            except BaseException:
                pass
            await matcher.finish(lang.text("chatgpt.cache_cleaned", [], event.user_id))
        elif argv[0] == "show":
            reply = ""
            cache = messages[str(event.group_id)]
            for item in cache[1:]:
                reply += f"\n{'User: ' if item['role'] == 'user' else 'XDbot: '}{item['content']}"
            reply = lang.text("chatgpt.cache", [reply], event.user_id)
            await matcher.finish(
                MessageSegment.reply(event.message_id) + MessageSegment.text(reply)
            )
    except BaseException:
        await _error.report(format_exc(), matcher)


def check_gpt():
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "hi"}]
        )
        return True
    except:
        return None


# [HELPSTART] Version: 2
# Command: gpt
# Usage: gpt <内容...>：与 XDbot2GPT 对话（20vi/次）
# Usage: gpt-config {apikey|proxy} <值>：配置 XDbot2GPT （不建议）
# Usage: gpt-cache {show|reset}：展示/重置 XDbot2GPT 会话缓存
# Usage: gpt-reset-as <内容>：以 <内容> 作为设定并重置会话（45vi/次）
# Info: XDbot2GPT
# Check: check_gpt
# [HELPEND]
