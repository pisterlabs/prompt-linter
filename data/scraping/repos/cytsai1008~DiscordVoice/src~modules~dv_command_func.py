import asyncio
import contextlib
import datetime
import os
import re
import time

import bs4
# import cloudscraper
import httpx
import metadata_parser
# import openai
import openai_async
from discord.ext import commands

import src.modules.dv_tool_function as tool_function
from src.modules import tts_func


async def content_convert(ctx, lang: str, locale: dict, content: str) -> str:
    content = await commands.clean_content(
        fix_channel_mentions=True, use_nicknames=True
    ).convert(ctx, content)

    # Emoji Replace
    if re.findall(r"<a?:[^:]+:\d+>", content):
        emoji_id = re.findall(r"<a?:[^:]+:\d+>", content)
        emoji_text = re.findall(r"<a?:([^:]+):\d+>", content)
        for i in range(len(emoji_id)):
            content = content.replace(
                emoji_id[i],
                tool_function.convert_msg(
                    locale,
                    lang,
                    "variable",
                    "say",
                    "emoji",
                    ["data_emoji", emoji_text[i]],
                ),
            )

    content = await _content_link_replace(content, lang, locale)
    return content


async def _get_web_title(client, url: str) -> (str, str):
    try:
        await tool_function.postgres_logging(f"Fetching web title: {url}")
        resp = await client.get(url)
        metadata = metadata_parser.MetadataParser(
            html=resp.text, search_head_only=False
        )
        title = metadata.get_metadatas("title")[0]
        # resp.headers.get('Server', '').startswith('cloudflare')
        if title.find("Attention Required!") != -1:
            resp = httpx.get(url, follow_redirects=True)
            metadata = metadata_parser.MetadataParser(
                html=resp.text, search_head_only=False
            )
            title = metadata.get_metadatas("title")[0]
        if title == "":
            soup = bs4.BeautifulSoup(resp.text, "lxml")
            title = soup.title.text
        return url, title
    except Exception:
        return url, ""


# noinspection HttpUrlsUsage
async def _content_link_replace(content: str, lang, locale: dict) -> str:
    """Return the head in the link if content has links"""

    # clear localhost 0.0.0.0 127.0.0.1
    local_list = [
        re.findall("(https?://127.0.0.1:\d{1,5}/?[^ ]+)", content),
        re.findall("(localhost:\d{1,5}/?[^ ]+)", content, flags=re.IGNORECASE),
        re.findall("(https?://0.0.0.0:\d{1,5}/?[^ ]+)", content),
        re.findall("(127.0.0.1:\d{1,5}/?[^ ]+)", content),
        re.findall("(0.0.0.1:\d{1,5}/?[^ ]+)", content),
    ]

    for i in local_list:
        for j in i:
            content = content.replace(j, "")

    web_regex = r"(https?://(?:www\.|(?!www))[a-zA-Z\d][a-zA-Z\d-]+[a-zA-Z\d]\.\S{2,}|www\.[a-zA-Z\d][a-zA-Z\d-]+[a-zA-Z\d]\.\S{2,}|https?://(?:www\.|(?!www))[a-zA-Z\d]+\.\S{2,}|www\.[a-zA-Z\d]+\.\S{2,})"

    if not re.findall(
        web_regex,
        content,
        flags=re.IGNORECASE,
    ):
        return content

    url = re.findall(
        web_regex,
        content,
        flags=re.IGNORECASE,
    )

    # remove duplicate
    url = list(set(url))
    if len(url) <= 3:
        headers = {
            "user-agent": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Googlebot/2.1; +http://www.google.com/bot.html) Chrome/115.0.5790.110 Safari/537.36"
        }

        new_url = []
        for i in url:
            if not i.startswith("http"):
                i = i.replace(i, f"http://{i}")
            new_url.append(i)

        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            tasks = [_get_web_title(client, i) for i in new_url]
            results = await asyncio.gather(*tasks)
            for i in results:
                convert_text = tool_function.convert_msg(
                    locale, lang, "variable", "say", "link", ["data_link", i[1]]
                )

                content = content.replace(i[0], convert_text)

    else:
        for i in url:
            content = content.replace(i, "")

    return content


async def check_voice_platform(
    user_platform_set: bool,
    user_id: str | int,
    guild_platform_set: bool,
    guild_id: str | int,
    lang: str,
) -> str:
    """Return the platform of the user or guild (default: Google)"""
    if (
        lang
        in tool_function.read_local_json("lang_list/google_languages.json")[
            "Support_Language"
        ]
        and lang
        not in tool_function.read_local_json("lang_list/azure_languages.json")[
            "Support_Language"
        ]
    ):
        return "Google"
    if (
        lang
        in tool_function.read_local_json("lang_list/azure_languages.json")[
            "Support_Language"
        ]
        and lang
        not in tool_function.read_local_json("lang_list/google_languages.json")[
            "Support_Language"
        ]
    ):
        return "Azure"
    user_id = f"user_{str(user_id)}"
    if (
        user_platform_set
        and tool_function.read_db_json("user_config")[user_id]["platform"] == "Google"
    ):
        await tool_function.postgres_logging("Init Google TTS API 1")
        return "Google"

    elif (
        user_platform_set
        and tool_function.read_db_json("user_config")[user_id]["platform"] == "Azure"
    ):
        await tool_function.postgres_logging("Init Azure TTS API 1")
        return "Azure"
    elif (
        guild_platform_set
        and tool_function.read_db_json(f"{guild_id}")["platform"] == "Google"
    ):
        await tool_function.postgres_logging("Init Google TTS API 2")
        return "Google"
    elif (
        guild_platform_set
        and tool_function.read_db_json(f"{guild_id}")["platform"] == "Azure"
    ):
        await tool_function.postgres_logging("Init Azure TTS API 2")
        return "Azure"
    elif not user_platform_set and not guild_platform_set:
        await tool_function.postgres_logging("Init Google TTS API 3")
        return "Google"
    else:
        await tool_function.postgres_logging(
            f"You found a bug\n"
            f"User platform: {user_platform_set}\n"
            f"User id: {user_id}\n"
            f"Guild platform: {guild_platform_set}\n"
            f"Guild id: {guild_id}\n"
        )
        return "Something wrong"


def name_convert(
    ctx, lang: str, locale: dict, content: str, gpt: None | bool = False
) -> str:
    if gpt:
        return tool_function.convert_msg(
            locale,
            lang,
            "variable",
            "say",
            "inside_said",
            [
                "user",
                "ChatGPT",
                "data_content",
                content,
            ],
        )
    user_id = ctx.author.id
    guild_id = ctx.guild.id

    try:
        username = ctx.author.display_name
    except AttributeError:
        username = ctx.author.name
    # get username length
    no_name = False
    send_time = int(
        time.mktime(datetime.datetime.now(datetime.timezone.utc).timetuple())
    )
    if tool_function.check_local_file(f"msg_temp/{guild_id}.json"):
        old_msg_temp = tool_function.read_local_json(f"msg_temp/{guild_id}.json")
        if old_msg_temp["1"] == user_id and send_time - int(old_msg_temp["0"]) <= 15:
            no_name = True
    id_too_long = False
    if len(username) > 20:
        if len(ctx.author.name) > 20:
            id_too_long = True
        else:
            username = ctx.author.name

    if id_too_long:
        username = tool_function.convert_msg(
            locale,
            lang,
            "variable",
            "say",
            "someone_name",
            None,
        )
        if ctx.author.voice is not None:
            content = tool_function.convert_msg(
                locale,
                lang,
                "variable",
                "say",
                "inside_said",
                [
                    "user",
                    username,
                    "data_content",
                    content,
                ],
            )
        else:
            content = tool_function.convert_msg(
                locale,
                lang,
                "variable",
                "say",
                "outside_said",
                [
                    "user",
                    username,
                    "data_content",
                    content,
                ],
            )
    elif not no_name:
        content = (
            tool_function.convert_msg(
                locale,
                lang,
                "variable",
                "say",
                "inside_said",
                [
                    "user",
                    username,
                    "data_content",
                    content,
                ],
            )
            if ctx.author.voice is not None
            else tool_function.convert_msg(
                locale,
                lang,
                "variable",
                "say",
                "outside_said",
                [
                    "user",
                    username,
                    "data_content",
                    content,
                ],
            )
        )
    return content


async def tts_convert(ctx, lang: str, content: str, platform_result: str) -> [bool]:
    guild_id = ctx.guild.id
    if platform_result == "Azure":
        await tool_function.postgres_logging("Init Azure TTS API")
        await tts_func.azure_tts_converter(content, lang, f"{guild_id}.mp3")
        return True

    elif platform_result == "Google":
        await tool_function.postgres_logging("Init Google TTS API")
        await tts_func.google_tts_converter(content, lang, f"{guild_id}.mp3")
        return True

    else:
        await tool_function.postgres_logging("Something Wrong")
        # send to owner
        await tts_func.google_tts_converter(content, lang, f"{guild_id}.mp3")
        return False


def is_banned(user_id: int | str, guild_id: int | str) -> bool:
    ban_list = tool_function.read_db_json("ban")
    if tool_function.check_dict_data(
        ban_list, f"{user_id}"
    ) and tool_function.check_dict_data(ban_list[f"{user_id}"], f"{guild_id}"):
        if int(ban_list[f"{user_id}"][f"{guild_id}"]["expire"]) >= int(
            time.mktime(datetime.datetime.now(datetime.timezone.utc).timetuple())
        ):
            return True
        with contextlib.suppress(Exception):
            del ban_list[f"{user_id}"][f"{guild_id}"]
        if not ban_list[f"{user_id}"]:
            with contextlib.suppress(Exception):
                del ban_list[f"{user_id}"]
        tool_function.write_db_json("ban", ban_list)
    return False


"""
async def gpt_process(lang: str, content: str) -> str:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are a friendly human, use {lang} to answer the question.",
            },
            {"role": "user", "content": content},
        ],
    )
    return completion["choices"][0]["message"]["content"]
    """


async def gpt_process(lang: str, content: str) -> str:
    response = await openai_async.chat_complete(
        os.environ["OPENAI_API_KEY"],
        timeout=20,
        payload={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": f"If no specific language is requested, use {lang} as a friendly human to respond. Otherwise, reply in the language requested by the user.",
                },
                {"role": "user", "content": content},
            ],
        },
    )
    return response.json()["choices"][0]["message"]["content"]
