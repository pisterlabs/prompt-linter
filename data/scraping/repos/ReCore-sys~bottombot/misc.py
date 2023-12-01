import datetime
import json
import os
import platform
import random
import time
import re

import asyncpraw as praw
import cpuinfo
import discord
import humanfriendly
import openai

try:
    import psutil  # type: ignore | So pylance will shut up

    psutilinstalled = True
except ModuleNotFoundError:
    psutilinstalled = False
import requests
from discord.ext import commands

import botlib
import bottomlib
import secretdata
import settings
import sqlbullshit
import utils

filepath = os.path.abspath(os.path.dirname(__file__))

sql = sqlbullshit.sql(filepath + "/data.db", "user")
openai.api_key = secretdata.openaikey

starttime = time.time()


reddit = praw.Reddit(
    client_id=secretdata.reddit_client_id,
    client_secret=secretdata.reddit_client_secret,
    user_agent=secretdata.reddit_user_agent,
)


class Misc(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def rewind(self, ctx, val=1):
        utils.log(ctx)
        if str(val).isnumeric():
            with open(f"{filepath}/serversettings/{ctx.guild.id}/replay.txt", "r") as f:
                lines = f.read().splitlines()
                last_line = lines[int(val) * -1]
                print(f"Mention: {last_line} was called")
                await ctx.send(last_line)

                with open(f"{filepath}/logs.txt", "a") as f:
                    f.write(
                        f"{datetime.datetime.now()} - {ctx.message.guild.name} | {ctx.message.author} : !rewind was called: result {val} was called with the result '{last_line}'\n"
                    )

        else:
            await ctx.send("I need a number stupid")

    @commands.Cog.listener()
    async def on_guild_leave(self, guild):
        os.system(f"rmdir settings/{guild.id}")
        with open(f"{filepath}/logs.txt", "a") as f:
            f.write(
                f"{datetime.datetime.now()}: Left a server! {guild.name} : {guild.id}\n"
            )
        print(f"Left a server! {guild.name} : {guild.id}")
        # this removes all the database stuff for when the bot leaves a server, whether it is kicked or the server is deleted.

    @commands.command()
    async def invite(self, ctx):
        utils.log(ctx)
        await ctx.send(
            "Server: https://discord.gg/2WaddNnuHh \nBot invite:  https://discord.com/api/oauth2/authorize?commands_id=758912539836547132&permissions=8&scope=bot"
        )

    @commands.command()
    async def code(self, ctx):
        utils.log(ctx)
        utils.log(ctx)
        await ctx.send(
            "Feel free to make commits and stuff.\nhttps://github.com/ReCore-sys/bottombot"
        )

    @commands.command()
    async def servers(self, ctx):
        utils.log(ctx)
        await ctx.send(len(list(commands.Bot.guilds)) + " servers have been infected!")

    @commands.command()
    async def roll(self, ctx, arg1):
        utils.log(ctx)
        if arg1.isnumeric():
            await ctx.send(random.randint(1, int(arg1)))
        else:
            await ctx.send("I need a number stupid")

    @commands.command()
    async def fox(self, ctx):
        utils.log(ctx)
        await ctx.send(f"https://randomfox.ca/images/{random.randint(1,122)}.jpg")

    @commands.command()
    async def pussy(self, ctx):
        utils.log(ctx)
        while True:
            URL = "https://aws.random.cat/meow"
            r = requests.get(url=URL)
            t = r.json()
            if ".mp4" in t["file"]:
                pass
            else:
                break
        await ctx.send(t["file"])

    @commands.command()
    async def dog(self, ctx):
        utils.log(ctx)
        URL = "https://random.dog/woof.json"
        r = requests.get(url=URL)
        t = r.json()
        await ctx.send(t["url"])

    @commands.command()
    async def xkcd(self, ctx):
        utils.log(ctx)
        URL = f"https://xkcd.com/{random.randint(1,2400)}/info.0.json"
        r = requests.get(url=URL)
        t = r.json()
        await ctx.send(t["img"])

    ttst = False

    @commands.command()
    async def tts(self, ctx, val=None):
        utils.log(ctx)
        global ttst
        if val is not None:
            if val == "on":
                ttst = True
            else:
                ttst = False
            await ctx.send(f"TTS set to {ttst}")
        else:
            await ctx.send("You need to give me input")

    @commands.command(alias=["bottombot"])
    async def bb(self, ctx, *, args):
        utils.log(ctx)
        if settings.check(ctx.message.guild.id, "get", "bb"):

            if botlib.check_banned(ctx):
                async with ctx.channel.typing():
                    tokens = len(args) / 4
                    if tokens >= 75:
                        await ctx.send(
                            "Sorry, that input was too big. Please try something smaller"
                        )
                        return
                    if sql.doesexist(ctx.author.id) is False:
                        await ctx.send(
                            "You don't have an economy account yet. Please make one with -bal"
                        )
                        return
                    if sql.get(ctx.author.id, "money") < 3:
                        await ctx.send("Sorry, you need more money to use the bot")
                    else:
                        sql.take(3, ctx.author.id, "money")
                        ctx.message.channel.typing()
                        with open(filepath + "/convfile.txt") as f:
                            # Here to see what I changed? Nothing. But shhh.
                            response = openai.Completion.create(
                                engine="curie",
                                prompt=(f.read()).format(args),
                                temperature=0.9,
                                max_tokens=100,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0.6,
                                stop=[" Human:", " AI:", "\n"],
                            )
                        bot = response.choices[0].text
                        print(
                            f"{datetime.datetime.now()} - {ctx.message.guild.name} | {ctx.message.author} : -bb: {args} -> {bot}\n"
                        )
                        with open(f"{filepath}/logs.txt", "a") as f:
                            f.write(
                                f"{datetime.datetime.now()} - {ctx.message.guild.name} | {ctx.message.author} : -bb: {args} -> {bot}\n"
                            )

                        await ctx.reply(bot)
            else:
                # there are a few people banned from using this command. These are their ids
                await ctx.reply(botlib.nope())
        else:
            await ctx.send("Sorry, the chatbot is disabled for this server")

    @commands.command()
    async def ping(self, ctx):
        utils.log(ctx)
        time_1 = time.perf_counter()
        await ctx.trigger_typing()
        time_2 = time.perf_counter()
        ping = round((time_2 - time_1) * 1000)
        embed = discord.Embed(
            title="Pong!",
            description=f"Currant Latancy = {ping}. Lol u got slow internet",
        )
        await ctx.send(embed=embed)
        print("Done: ping = " + str(ping))

    @commands.command()
    async def info(self, ctx):
        utils.log(ctx)
        embed = discord.Embed(
            title="Bottombot",
            description="If it works let me know. I'd be pretty suprised.",
            color=0x8800FF,
        )
        embed.add_field(
            name="Creator", value="<@451643725475479552>", inline=True)
        embed.add_field(
            name="Reason", value="I was bored and want to make a bot", inline=False
        )
        embed.add_field(name="Functionality?", value="no", inline=False)
        embed.add_field(
            name="Made with",
            value="Love, care, ages on stackoverflow.com, bugging <@416101701574197270> and copious amounts of cocaine.",
            inline=False,
        )
        print("")

        await ctx.send(embed=embed)

    @commands.command()
    async def cease(self, ctx):
        utils.log(ctx)
        if ctx.message.author.id == 451643725475479552:
            exit()
        else:
            # command to turn off the bot. Only I can use it.
            await ctx.send("Lol nah")

    @commands.command()
    async def reboot(self, ctx):
        utils.log(ctx)
        if ctx.message.author.id == 451643725475479552:
            print("Rebooting ")
            os.system(f"python {filepath}/bot.py")
            # os.kill("java.exe")
            # os.kill("cmd.exe")
            exit()
        else:
            # command to reboot the bot. Only I can use it.
            await ctx.send("Lol nah")

    pingC = None
    pingU = None

    @commands.command()
    async def trans(self, ctx):
        utils.log(ctx)
        await ctx.send(
            ":transgender_flag: Trans rights are human rights :transgender_flag: "
        )

    @commands.command()
    async def cf(self, ctx):
        utils.log(ctx)
        await ctx.send(random.choice(["Heads", "Tails"]))

    @commands.command()
    async def updates(self, ctx, remove=False):
        utils.log(ctx)
        channel = ctx.message.channel.id
        with open(filepath + "/json/configs.json", "r") as f:
            j = json.load(f)
            channels = j["updates"]
            if channel in channels:
                await ctx.send(
                    "Ok, removed this channel from the list to recieve updates"
                )
                channels.remove(channel)
            else:
                await ctx.send("Ok, added this channel to the list to recieve updates")
                channels.append(channel)
            j["updates"] = channels
        with open(filepath + "/json/configs.json", "w") as f:
            json.dump(j, f)

    @commands.command()
    async def update(self, ctx, *, args):
        utils.log(ctx)
        if ctx.message.author.id == 451643725475479552:
            with open(filepath + "/json/configs.json", "r") as f:
                servers = json.load(f)
                for x in servers["updates"]:
                    address = self.bot.get_channel(int(x))
                    await address.send("**ANNOUNCMENT**")
                    await address.send(args)
                    time.sleep(0.5)

    @commands.command()
    async def duck(self, ctx):
        utils.log(ctx)
        await ctx.send(
            "http://hd.wallpaperswide.com/thumbs/duck_3-t2.jpg"
        )

    @commands.command()
    async def bottomgear(self, ctx):
        utils.log(ctx)
        global ttst
        output = bottomlib.bottomchoice()
        await ctx.send(output)
        print("Done: -bottomgear")
        with open(f"{filepath}/logs.txt", "a") as f:
            f.write(
                f"{datetime.datetime.now()} - {ctx.message.guild.name} | {ctx.message.author} : -bottomgear {output}\n"
            )

    @commands.command()
    async def help(self, ctx, menu=None):
        utils.log(ctx)
        result = json.load(open(f"{filepath}/json/help.json"))
        result2 = {}
        bannedhelps = []
        for x in result:
            list = []
            if x not in bannedhelps:
                for v in result[x]:
                    if v not in bannedhelps:
                        result2[v] = result[x][v]
        if menu not in result2:
            embed = discord.Embed(
                title="Help",
                description="Welcome to the help menu. Do -help <command> to see what an individual command does",
                color=0x1E00FF,
            )
            for x in result:
                list = []
                for v in result[x]:
                    list.append(v)
                nicelist = ", ".join(list)
                embed.add_field(name=x, value=f"`{nicelist}`", inline=True)

            await ctx.send(embed=embed)
        else:
            embed = discord.Embed(
                title=menu, description=result2[menu], color=0x1E00FF)
            await ctx.send(embed=embed)

    @commands.command()
    async def stats(self, ctx):
        utils.log(ctx)
        uname = platform.uname()
        cputype = cpuinfo.get_cpu_info()["brand_raw"]
        osversion = uname.version
        ostype = uname.system
        uptime = time.time() - starttime
        if psutilinstalled:
            cores = psutil.cpu_count(logical=True)
            cpuuse = psutil.cpu_percent()
            svmem = psutil.virtual_memory()
            mem = utils.convert_bytes(svmem.total)
            used = utils.convert_bytes(svmem.used)
            percent = svmem.percent
            partition = psutil.disk_partitions()[0]
            partition_usage = psutil.disk_usage(partition.mountpoint)
            disk_total = utils.convert_bytes(partition_usage.total)
            disk_used = utils.convert_bytes(partition_usage.used)
            disk_percent = partition_usage.percent
        embed = discord.Embed(
            title="Stats", description="System stats", color=0x1E00FF)
        embed.add_field(
            name="Uptime",
            value=f"{humanfriendly.format_timespan(uptime)}",
            inline=False,
        )
        embed.add_field(
            name="CPU",
            value=f"{cputype} {f'({cores} cores)' if psutilinstalled == True else ''}",
            inline=False,
        )
        if psutilinstalled:
            embed.add_field(name="CPU Usage", value=f"{cpuuse}%", inline=False)
            embed.add_field(
                name="OS", value=f"{ostype} ({osversion})", inline=False)
            embed.add_field(name="Memory", value=f"{mem}", inline=False)
            embed.add_field(
                name="Used", value=f"{used} ({percent}%)", inline=False)
            embed.add_field(name="Disk Total",
                            value=f"{disk_total}", inline=False)
            embed.add_field(
                name="Disk Used", value=f"{disk_used} ({disk_percent}%)", inline=False
            )
        await ctx.send(embed=embed)

    @commands.command()
    async def guide(self, ctx):
        utils.log(ctx)
        with open(f"{filepath}/static/guide.txt", "r") as f:
            guide = f.read()
            user = self.bot.get_user(ctx.message.author.id)
            await user.send(guide)

    @commands.command()
    async def feedback(self, ctx, *, args):
        utils.log(ctx)
        await ctx.send("Ok, your feedback was sent!")
        me = self.bot.get_user(451643725475479552)
        await me.send(
            f"{ctx.message.author.name} has sent the following feedback: \n{args}"
        )

    async def getmedia(self):
        subreddits = botlib.configs("misc", "subreddits")
        sub = await reddit.subreddit(random.choice(subreddits))
        post = await sub.random()

        if post is None:
            modes = ["hot", "new", "rising", "top"]
            mode = random.choice(modes)
            if mode == "hot":
                post = sub.hot()
            elif mode == "new":
                post = sub.new()
            elif mode == "rising":
                post = sub.rising()
            elif mode == "top":
                post = sub.top()
            choices = []
            counter = 0
            async for x in post:
                choices.append(x)
                if counter == 10:
                    break
                counter += 1
            post = random.choice(choices)
        author = post.author.name
        title = post.title
        if post.media is not None:
            media = post.media[list((post.media).keys())[
                0]]["scrubber_media_url"]
        else:
            media = post.url
        return author, media, post, title

    redditstuff = None

    @commands.command()
    async def shitpost(self, ctx):
        utils.log(ctx)
        if self.redditstuff is None:
            author, media, post, title = await self.getmedia()
            self.redditstuff = author, media, post, title
            await ctx.send(
                f"Courtesy of u/{author} on {post.subreddit_name_prefixed}\n\n**{title}**"
            )
            await ctx.send(media)
            author, media, post, title = await self.getmedia()
            self.redditstuff = author, media, post, title
        else:
            author, media, post, title = self.redditstuff
            await ctx.send(
                f"Courtesy of u/{author} on {post.subreddit_name_prefixed}\n\n**{title}**"
            )
            await ctx.send(media)
            self.redditstuff = await self.getmedia()

    @commands.Cog.listener()
    async def on_message(self, ctx):
        # TODO: Add this to a json file somehow
        kms = [
            re.compile(r".*kill\s*myself.*", re.IGNORECASE),
            re.compile(r".*end\s*myself.*", re.IGNORECASE),
            re.compile(r".*end\s*my\s*life.*", re.IGNORECASE),
            re.compile(r".*commit\s*die.*", re.IGNORECASE),
            re.compile(r".*wish\s*i\s*was\s*dead.*", re.IGNORECASE),
            re.compile(r".*want\s*to\s*die.*", re.IGNORECASE),
            re.compile(r".*shoot\s*myself.*", re.IGNORECASE),
            re.compile(r".*kys.*", re.IGNORECASE),
            re.compile(r".*kms.*", re.IGNORECASE),
            re.compile(r".*ima\s*kms.*", re.IGNORECASE),
            re.compile(r".*just\s*kms.*", re.IGNORECASE),
            re.compile(r".*commit\s*die.*", re.IGNORECASE)
        ]

        cont = ctx.clean_content
        cont = cont.replace("wanna", "want to").replace("imma", "ima")
        for x in kms:

            if utils.match_regex(x, cont):
                embed = discord.Embed(title="You are not alone",
                                      description="Your life is important. We all care very deeply about you. I understand you don't feel like you matter right know, but I can tell you with 100% confidence that you do. I know you might be reluctant, but please just give the suicide prevention hotline just one more chance.",)
                embed.set_thumbnail(
                    url="https://www.edinburghhsc.scot/wp-content/uploads/2021/04/Suicide_website_icon.png")
                embed.add_field(name="United States",
                                value="Call (800) 2738255")
                embed.add_field(name="Australia",
                                value="Call 131114", inline=False)
                embed.add_field(name="United Kingdom",
                                value="Call 116123", inline=False)
                embed.add_field(
                    name="Canada", value="Call 18334564566 or text 45645", inline=False)
                embed.add_field(
                    name="India", value="Call 18005990019", inline=False)
                embed.add_field(
                    name="Japan", value="Call 810352869090", inline=False)
                embed.add_field(
                    name="Other countries", value="https://www.opencounseling.com/suicide-hotlines", inline=False)
                embed.set_footer(
                    text="I care about you. Please try to give the helplines just one chance. I know you can make it through this.")
                embed.set_author(name="Inspired by Suicide prevention bot",
                                 url="https://github.com/Bobrobot1/Suicide-Prevention-Bot/")
                await ctx.channel.send(embed=embed)
                break

    @commands.command()
    async def suicide(self, ctx):
        embed = discord.Embed(title="You are not alone",
                              description="Your life is important. We all care very deeply about you. I understand you don't feel like you matter right know, but I can tell you with 100% confidence that you do. I know you might be reluctant, but please just give the suicide prevention hotline just one more chance.",)
        embed.set_thumbnail(
            url="https://www.edinburghhsc.scot/wp-content/uploads/2021/04/Suicide_website_icon.png")
        embed.add_field(name="United States",
                        value="Call (800) 2738255")
        embed.add_field(name="Australia",
                        value="Call 131114", inline=False)
        embed.add_field(name="United Kingdom",
                        value="Call 116123", inline=False)
        embed.add_field(
            name="Canada", value="Call 18334564566 or text 45645", inline=False)
        embed.add_field(
            name="India", value="Call 18005990019", inline=False)
        embed.add_field(
            name="Japan", value="Call 810352869090", inline=False)
        embed.add_field(
            name="Other countries", value="https://www.opencounseling.com/suicide-hotlines", inline=False)
        embed.set_footer(
            text="I care about you. Please try to give the helplines just one chance. I know you can make it through this.")
        embed.set_author(name="Inspired by Suicide prevention bot",
                         url="https://github.com/Bobrobot1/Suicide-Prevention-Bot/")
        await ctx.channel.send(embed=embed)


def setup(bot):
    bot.add_cog(Misc(bot))
