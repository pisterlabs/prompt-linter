import aiohttp
import asyncio
import pykakasi
import discord
import asyncio
import random
import openai
import io

from utils import show_error
from discord.ext import commands, menus


def format_jp_entry(entry):
    try:
        return f"{entry['word']}【{entry['reading']}】"
    except KeyError:
        try:
            return entry["reading"]
        except KeyError:
            try:
                return entry["word"]
            except KeyError:
                return "???"

class DictSource(menus.ListPageSource):
    def __init__(self, data):
        super().__init__(data, per_page=1)

    async def format_page(self, menu, entry):
        e = discord.Embed(
            title = f"Result #{menu.current_page + 1}",
            description = format_jp_entry(entry['japanese'][0])
        )
        if tags := [
            *["common"]*entry.get("is_common", False),
            *sorted(f"JLPT {x.partition('-')[2]}" for x in entry.get("jlpt", []))[-1:],
        ]:
            e.title += f" ({', '.join(tags)})"
        for i, sense in enumerate(entry["senses"], start=1):
            e.add_field(
                name = ", ".join(sense["parts_of_speech"]) if sense["parts_of_speech"] else "\u200b",
                value = " | ".join([
                    f"{i}. " + "; ".join(sense["english_definitions"]),
                    *filter(None, [", ".join(f"*{x}*" for x in sense["tags"] + sense["info"])]),
                ]),
                inline=False
            )
        if len(entry["japanese"]) > 1:
            e.add_field(name = "Other forms", value = "\n".join(format_jp_entry(x) for x in entry["japanese"][1:]), inline=False)
        return e

class Japanese(commands.Cog):
    """Weeb stuff."""

    def __init__(self, bot):
        self.bot = bot
        kakasi = pykakasi.kakasi()
        kakasi.setMode("H", "a")
        kakasi.setMode("K", "a")
        kakasi.setMode("J", "a")
        kakasi.setMode("s", True)
        self.conv = kakasi.getConverter()

    @commands.command(aliases=["ro", "roman", "romanize", "romanise"])
    async def romaji(self, ctx, *, text: commands.clean_content = None):
        """Romanize Japanese text."""
        await ctx.send(self.conv.do(text or self.last_lyric_msg))

    @commands.command(aliases=["jp", "jsh", "dictionary", "dict"])
    async def jisho(self, ctx, *, query):
        """Look things up in the Jisho dictionary."""
        async with self.bot.session.get("https://jisho.org/api/v1/search/words", params={"keyword": query}) as resp:
            if resp.status == 200:
                data = await resp.json()
            else:
                data = None
        if not data["data"]:
            return await show_error(ctx, "That query returned no results.")
        pages = menus.MenuPages(source=DictSource(data["data"]), clear_reactions_after=True)
        await pages.start(ctx)

    @commands.command(aliases=["jatrans", "transja", "jtrans", "jptrans", "transjp", "transj", "tj", "jtr", "jpt", "jt",
                               "whatdidlyricjustsay", "what'dlyricsay", "whtdlysay", "wdls", "wls", "what",
                               "weebtrans", "weebt", "deweeb", "unweeb", "transweeb", "tweeb", "tw",
                               ";)", "forumbra", "inadequateweeb", "inadqweeb", "otherlanguagesscareme",
                               "otherlangsscareme", "that'snotenglish", "notenglish", "noen", "日本語から",
                               "ifyouhaveajapaneseimewhyareyouusingashittygoogletranslatecommand", "ifuhvajpimeyruusingshitgtcmd"])
    async def jatranslate(self, ctx, *, lyric_quote: commands.clean_content = None):
        """Translate Japanese."""
        if not lyric_quote:
            messages = [m async for m in ctx.history(limit=10) if not m.content.startswith("!") and not m.author.bot]
            p = "\n".join([f"{i}: {m.content}" for i, m in enumerate(messages)][::-1])
            completion = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a bot whose purpose is to identify which message from a list of different messages is the "most Japanese".
You should prioritize actual Japanese text, but after that you may take into consideration cultural references or references to anime and manga.
The messages will be numbered, and you must simply say the number of which message is the most Japanese. Say nothing else besides the number on its own. If no message is remotely Japanese at all, then say "nil"."""},
                    {"role": "user", "content": """2: are you a weeb
1: 分からないんだよ
0: got it"""},
                    {"role": "assistant", "content": "1"},
                    {"role": "user", "content": """3: if only it was possible to look out the window on a plane
2: olivia is definitely in on this
1: why else japan !!?!
0: そうそう"""},
                    {"role": "assistant", "content": "0"},
                    {"role": "user", "content": """4: fastest transition in the west
3: it would be so funny if xenia was real
2: it would
1: wish i were real
0: too bad xenia is a cat walking on a keyboard with a predictive wordfilter applied"""},
                    {"role": "assistant", "content": "nil"},
                    {"role": "user", "content": """4: me fr
3: do you think they would let me take blåhaj on the plane if I went to coral
2: 変カャット
1: wtf
0: oh"""},
                    {"role": "assistant", "content": "2"},
                    {"role": "user", "content": """2: nooo
1: mjauuu
0: wooo"""},
                    {"role": "assistant", "content": "nil"},
                    {"role": "user", "content": """5: Also I refuse to make a non-gc language
4: Other than Forth
3: So no update
2: the nail that sticks out will get hammered down
1: no fucking way
0: getting closer"""},
                    {"role": "assistant", "content": "2"},
                    {"role": "user", "content": p},
                ],
            )
            r = completion["choices"][0]["message"]["content"]
            if not r.isdigit() or int(r) not in range(len(messages)):
                return await ctx.send("I don't see anything to translate.")
            msg = messages[int(r)]
            lyric_quote = msg.content
        else:
            msg = ctx.message
        completion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a translator whose job is to determine what language something is written in. The only things you ever say are "Japanese" and "Not Japanese".
Even if something is a direct reference to a phrase in Japanese, if it is not literally written in Japanese, you always say "Not Japanese"."""},
                {"role": "user", "content": lyric_quote},
            ],
        )
        if "not" in completion["choices"][0]["message"]["content"].lower():
            prompt = "You are a helpful translator. When given a reference to Japanese culture or media, you explain the reference briefly but comprehensively, in English."
        else:
            prompt = """If you are given text that is entirely or partially written in Japanese, you provide a translation of the text in English.
When translating, you never give additional commentary or explanations; you only give the literal translation of the text and nothing else.
Your responses never contain the text "Translation:"."""
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": lyric_quote},
            ],
        )
        result = completion["choices"][0]["message"]["content"]
        if len(result) > 2000:
            await msg.reply(file=discord.File(io.StringIO(result), "resp.txt"))
        else:
            await msg.reply(result)


async def setup(bot):
    await bot.add_cog(Japanese(bot))
