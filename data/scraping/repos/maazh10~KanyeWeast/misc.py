import asyncio
import json
import random

import discord
import openai
import requests
from discord.ext import commands

from cogs.dev import DevelopersOnly
from cogs.utils import UserBanned, get_quote


class Miscellaneous(commands.Cog):
    """Rando stuff."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        with open("secrets.json", "r") as f:
            self.keys = json.load(f)

    ##############################
    ####### COG BAN CHECK ########
    ##############################

    async def cog_check(self, ctx: commands.Context) -> bool:
        dev = self.bot.get_cog("DevelopersOnly")
        assert isinstance(dev, DevelopersOnly)
        if ctx.author.id in dev.banned_set:
            raise UserBanned(ctx.message.author)
        return True

    ##############################
    ##############################
    ##############################

    @commands.command(
        name="hello",
        brief="Says hello",
        help="Says hello to whoever used the `hello` command",
    )
    async def hello(self, ctx: commands.Context):
        if await self.bot.is_owner(ctx.author):
            if ctx.message.content.endswith("son"):
                await ctx.send("Hello master!")
            else:
                await ctx.send(f"Hello {ctx.author.display_name}!")
        else:
            if ctx.message.content.endswith("dad"):
                await ctx.send("Hello son!")
            else:
                await ctx.send(f"Hello {ctx.author.display_name}!")

    @commands.command(
        aliases=["donda?", "donda"],
        brief="Says whether Donda is out or not",
        help="Says whether Donda is out or not (long version)",
    )
    async def rembr(self, ctx: commands.Context):
        await ctx.send("i rember <:3736_GalaxyBrainPepe:769650997681061950>")
        embed = discord.Embed()
        embed.description = (
            "[Out 2021](https://music.apple.com/us/album/donda/1583449420)"
        )
        await ctx.send(embed=embed)

    @commands.command(
        name="quote",
        brief="Says a Kanye quote",
        help="Says a Kanye quote using a Kanye quote API",
    )
    async def quote(self, ctx: commands.Context):
        embed = discord.Embed()
        assert self.bot.user is not None
        assert self.bot.user.avatar is not None
        embed.color = 3348751
        embed.set_author(name="Kanye West", icon_url=self.bot.user.avatar.url)

        embed.description = (
            f"[{get_quote()}](https://www.youtube.com/watch?v=dQw4w9WgXcQ)"
        )
        await ctx.send(embed=embed)

    @commands.command(
        name="morning",
        brief="Kanye says good morning",
        help="Are you dumb what is there to understand",
    )
    async def morning(self, ctx: commands.Context):
        embed = discord.Embed()
        embed.set_image(
            url="https://tenor.com/view/alarm-wake-up-tired-so-gif-24728280.gif"
        )
        await ctx.send(embed=embed)

    @commands.command(
        name="rick", brief="Get Rick'd", help="Are you dumb what is there to understand"
    )
    async def rick(self, ctx: commands.Context):
        embed = discord.Embed()
        embed.set_image(
            url="https://media1.tenor.com/images/3e30fa16b8a79b44185060d0df450009/tenor.gif?itemid=19920902"
        )
        embed.description = "Never gonna give you up"
        await ctx.send(embed=embed, delete_after=8)

    @commands.command(
        name="fuck",
        brief="fuck",
        help="fuck",
    )
    async def fuck(self, ctx: commands.Context, *fucks):
        if ctx.message.mentions:
            for user in ctx.message.mentions:
                for _ in fucks:
                    await ctx.send(f"{user.mention} fuck")
            return
        if not fucks:
            fucks = [0]
        for _ in fucks:
            await ctx.send("fuck me")

    @commands.command(
        name="joke",
        brief="Sends a joke",
        help="Sends a joke through jokeAPI. Do &joke {category} for a specific category. Choose from [programming, misc, dark, pun, spooky, christmas]. If no category is specified a random one will be picked.",
    )
    async def joke(self, ctx, *args):
        categories = ["programming", "misc", "dark", "pun", "spooky", "christmas"]
        category = "Any"
        if len(args) == 1:
            if args[0] in categories:
                category = args[0].lower()
            else:
                await ctx.send("Invalid category")
                return
        elif len(args) > 1:
            await ctx.send("Too many arguements")
            return
        response = requests.get(f"https://v2.jokeapi.dev/joke/{category}")
        json_data = json.loads(response.text)
        if json_data["type"] == "twopart":
            setup = json_data["setup"]
            punchline = json_data["delivery"]
            await ctx.send(setup + "\n" + "||" + punchline + "||")
            return
        setup = json_data["joke"]
        await ctx.send(setup)

    @commands.command(
        name="",
        brief="Sends a cute animal pic.",
        help="Sends a cute animal pic through an API. Do &animal list for a list of animals. If no animal is specified a random one will be picked.",
    )
    async def animal(self, ctx: commands.Context, animal: str = ""):
        animals = [
            "bird",
            "cat",
            "dog",
            "fox",
            "kangaroo",
            "koala",
            "panda",
            "raccoon",
            "whale",
            "seal",
            "duck",
            "whale",
        ]
        if not animal:
            animal = random.choice(animals)
        if animal == "list":
            msg = "```\n"
            for animal in animals:
                msg += animal + "\n"
            msg += "```"
            await ctx.send(msg)
            return
        if animal not in animals:
            await ctx.send(f"No pics for {animal}")
            return
        embed = discord.Embed(title=animal.capitalize(), color=discord.Colour.random())
        if animal == "seal":
            seal_num = random.randint(0, 84)
            embed.set_image(
                url=f"https://raw.githubusercontent.com/FocaBot/random-seal/master/seals/{seal_num:04}.jpg"
            )
        elif animal == "duck":
            r = requests.get("https://random-d.uk/api/v2/random")
            data = r.json()
            embed.set_image(url=data["url"])
        elif animal == "whale":
            r = requests.get("https://some-random-api.ml/img/whale")
            data = r.json()
            embed.set_image(url=data["link"])
        else:
            r = requests.get(f"https://some-random-api.ml/animal/{animal}")
            data = r.json()
            embed.set_image(url=data["image"])
            embed.set_footer(text=data["fact"])
        await ctx.send(embed=embed)

    @commands.command(
        name="",
        brief="Send user's avatar back with overlay.",
        help="Send user's avatar back with overlay. Do &overlay list for a list of overlays. If no overlay is specified a random one will be picked.",
    )
    async def overlay(
        self, ctx: commands.Context, overlay: str = "", user: discord.User = None
    ):
        overlays = ["comrade", "gay", "glass", "jail", "passed", "triggered", "wasted"]
        if not overlay:
            overlay = random.choice(overlays)
        if overlay == "list":
            msg = "```\n"
            for overlay in overlays:
                msg += overlay + "\n"
            msg += "```"
            await ctx.send(msg)
            return
        if overlay not in overlays:
            await ctx.send(f"No overlay for {overlay}")
            return
        avatar = user.avatar.url if user else ctx.author.avatar.url
        await ctx.send(
            f"https://some-random-api.ml/canvas/overlay/{overlay}?avatar={avatar}"
        )

    @commands.command(
        name="",
        brief="Sends user's avatar back with a canvas.",
        help="Sends user's avatar back with a canvas. Do &canvas list for a list of canvases. If no canvas is specified a random one will be picked.",
    )
    async def canvas(
        self, ctx: commands.Context, canvas: str = "", user: discord.User = None
    ):
        canvases = [
            "bisexual",
            "blur",
            "circle",
            "heart",
            "horny",
            "its-so-stupid",
            "jpg",
            "lesbian",
            "lgbt",
            "lolice",
            "nonbinary",
            "pansexual",
            "pixelate",
            "simpcard",
            "spin",
            "tonikawa",
            "transgender",
        ]
        if not canvas:
            canvas = random.choice(canvases)
        if canvas == "list":
            msg = "```\n"
            for canvas in canvases:
                msg += canvas + "\n"
            msg += "```"
            await ctx.send(msg)
            return
        if canvas not in canvases:
            await ctx.send(f"No canvas for {canvas}")
            return
        try:
            avatar = user.avatar.url if user else ctx.author.avatar.url
        except AttributeError:
            avatar = user.default_avatar.url if user else ctx.author.default_avatar.url
        await ctx.send(
            f"https://some-random-api.ml/canvas/misc/{canvas}?avatar={avatar}"
        )

    @commands.command(
        name="donate",
        brief="Sends a link to donate to the bot.",
        help="Sends a link to donate to the bot.",
    )
    async def donate(self, ctx: commands.Context):
        await ctx.send("https://www.buymeacoffee.com/maazandbenny")

    @commands.command(
        name="",
        brief="Answers queries using GPT-3.5-turbo model.",
        help="Answers queries using GPT-3.5-turbo model i.e. chatGPT from OpenAI. You can use this to ask questions, get advice, or just have a conversation with the bot. Use &gpt {query} to get a response, to continue a conversation with previous context just reply to the bot's message with further queries. You have 120 seconds to reply to the bot's message.",
    )
    async def gpt(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        msg_context: list = [],
        cur_msg: discord.Message = None,
    ):
        msg_context = msg_context + [{"role": "user", "content": prompt}]
        cur_msg = cur_msg or ctx.message
        print(msg_context)
        async with ctx.typing():
            try:
                openai.api_key = self.keys["OPENAI_API_KEY"]
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=msg_context,
                )
                response = completion.choices[0].message.content
                if len(response) > 2000:
                    msg = await cur_msg.reply(response[:2000])
                    for i in range(2000, len(response), 2000):
                        await ctx.send(response[i : i + 2000])
                else:
                    msg = await cur_msg.reply(response)
            except openai.error.RateLimitError:
                await ctx.send("API rate limit exceeded.")
                return
        check = (
            lambda m: m is not None
            and m.channel == ctx.channel
            and m.reference is not None
            and m.reference.message_id == msg.id
            and m.author == ctx.author
        )
        reply = None
        try:
            reply = await self.bot.wait_for("message", check=check, timeout=120.0)
        except asyncio.TimeoutError:
            pass
        if reply:
            await ctx.invoke(
                self.gpt, prompt=reply.content, msg_context=msg_context, cur_msg=reply
            )

    @commands.command(
        name="",
        brief="Sends a random meme.",
        help="Sends a random meme.",
    )
    async def meme(self, ctx: commands.Context):
        r = requests.get("https://meme-api.com/gimme")
        data = r.json()
        embed = discord.Embed(
            title=data["title"],
            url=data["postLink"],
            color=discord.Colour.random(),
        )
        embed.set_image(url=data["url"])
        embed.set_footer(text=f"👍 {data['ups']} | 🙎 {data['author']}")
        await ctx.send(embed=embed)

    @commands.command(
        name="",
        brief="Sends a random greentext.",
        help="Sends a random greentext.",
    )
    async def greentext(self, ctx: commands.Context):
        r = requests.get("https://meme-api.com/gimme/greentext")
        data = r.json()
        embed = discord.Embed(
            title=data["title"],
            url=data["postLink"],
            color=discord.Colour.green(),
        )
        embed.set_image(url=data["url"])
        embed.set_footer(text=f"👍 {data['ups']} | 🙎 {data['author']}")
        await ctx.send(embed=embed)

    @commands.command(
        name="bb",
        brief="Sends a random Breaking Bad quote.",
        help="Sends a random Breaking Bad quote.",
    )
    async def breaking_bad(self, ctx: commands.Context):
        res = requests.get("https://api.breakingbadquotes.xyz/v1/quotes")
        data = res.json()
        embed = discord.Embed(color=discord.Colour.green())
        embed.description = data[0]["quote"]
        embed.set_footer(text=data[0]["author"])
        embed.set_thumbnail(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Breaking_Bad_logo.svg/800px-Breaking_Bad_logo.svg.png"
        )
        await ctx.send(embed=embed)

    @commands.command(
        name="YesOrNo",
        brief="Sends a random yes or no.",
        help="Sends a random yes or no.",
    )
    async def yesorno(self, ctx: commands.Context):
        res = requests.get("https://yesno.wtf/api")
        data = res.json()
        embed = discord.Embed(color=discord.Colour.random())
        embed.description = data["answer"]
        embed.set_image(url=data["image"])
        await ctx.send(embed=embed)


async def setup(bot: commands.Bot):
    await bot.add_cog(Miscellaneous(bot))
