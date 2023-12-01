import asyncio
import io
import os
import random
import sys
import tempfile

from typing import Literal, Optional

import discord
import openai
import uwupy  # type: ignore

from aiofiles import open as async_open
from discord.ext import commands

from helpers import checks, functions, regex, views, wock


class fun(commands.Cog, name="Fun"):
    def __init__(self, bot):
        self.bot: wock.wockSuper = bot
        openai.api_key = self.bot.config["api"]["openai"]
        self.eightball_responses = {
            "As I see it, yes": True,
            "Better not tell you now": False,
            "Concentrate and ask again": False,
            "Don't count on it": False,
            "It is certain": True,
            "It is decidedly so": True,
            "Most likely": True,
            "My reply is no": False,
            "My sources say no": False,
            "Outlook good": True,
            "Outlook not so good": False,
            "Reply hazy, try again": False,
            "Signs point to yes": True,
            "Very doubtful": False,
            "Without a doub.": True,
            "Yes": True,
            "Yes, definitely": True,
            "You may rely on it": True,
            "Ask again later": False,
            "I can't predict now": False,
        }

    # @commands.Cog.listener("on_user_message")
    # async def save_wordcloud(self, ctx: wock.Context, message: discord.Message):
    #     """Save message content to metrics in order to use wordcloud"""

    #     if not message.content:
    #         return

    #     if message.author.id in self.bot.owner_ids:
    #         return

    #     await self.bot.db.execute(
    #         "INSERT INTO metrics.messages VALUES ($1, $2, $3, $4, $5)",
    #         message.guild.id,
    #         message.channel.id,
    #         message.author.id,
    #         message.content,
    #         message.created_at,
    #     )

    @commands.command(name="uwu", usage="(text)", example="hello mommy", aliases=["uwuify"])
    async def uwu(self, ctx: wock.Context, *, text: str):
        """UwUify text"""

        await ctx.reply(uwupy.uwuify_str(text), allowed_mentions=discord.AllowedMentions.none())

    @commands.command(name="coinflip", usage="<heads/tails>", example="heads", aliases=["flipcoin", "cf", "fc"])
    async def coinflip(self, ctx: wock.Context, *, side: Literal["heads", "tails"] = None):
        """Flip a coin"""

        await ctx.load(f"Flipping a coin{f' and guessing **:coin: {side}**' if side else ''}..")
        await asyncio.sleep(1)

        coin = random.choice(["heads", "tails"])
        await getattr(ctx, ("approve" if (not side or side == coin) else "warn"))(
            f"The coin landed on **:coin: {coin}**" + (f", you **{'won' if side == coin else 'lost'}**!" if side else "!")
        )

    @commands.command(name="roll", usage="(sides)", example="6", aliases=["dice"])
    async def roll(self, ctx: wock.Context, sides: int = 6):
        """Roll a dice"""

        await ctx.load(f"Rolling a **{sides}-sided** dice..")
        await asyncio.sleep(1)

        await ctx.approve(f"The dice landed on **🎲 {random.randint(1, sides)}**")

    @commands.command(name="8ball", usage="(question)", example="am I pretty?", aliases=["8b"])
    async def eightball(self, ctx: wock.Context, *, question: str):
        """Ask the magic 8ball a question"""

        await ctx.load("Shaking the **magic 8ball**..")
        await asyncio.sleep(1)

        shakes = random.randint(1, 5)
        response = random.choice(list(self.eightball_responses.keys()))
        await getattr(ctx, ("approve" if self.eightball_responses[response] else "warn"))(
            f"After {functions.plural(shakes, code=True):shake} - **{response}**"
        )

    @commands.command(
        name="transparent",
        usage="(image)",
        example="dscord.com/chnls/999/..png",
        parameters={
            "alpha": {
                "require_value": False,
                "description": "Apply Alpha Matting to the image",
                "aliases": ["mask"],
            }
        },
        aliases=["tp"],
    )
    @checks.donator()
    @commands.cooldown(1, 10, commands.BucketType.user)
    @commands.max_concurrency(1, commands.BucketType.user)
    async def transparent(self, ctx: wock.Context, *, image: wock.ImageFinderStrict = None):
        """Remove the background of an image"""

        image = image or await wock.ImageFinderStrict.search(ctx)

        async with ctx.typing():
            response = await self.bot.session.get(image)
            if sys.getsizeof(response.content) > 15728640:
                return await ctx.warn("Image is too large to make **transparent** (max 15MB)")

            image = await response.read()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(
                    temp_dir, f"file{functions.hash(str(response.url))}." + regex.IMAGE_URL.match(str(response.url)).group("mime")
                )
                temp_file_output = os.path.join(
                    temp_dir, f"file{functions.hash(str(response.url))}_output." + regex.IMAGE_URL.match(str(response.url)).group("mime")
                )
                async with async_open(temp_file, "wb") as file:
                    await file.write(image)

                try:
                    terminal = await asyncio.wait_for(
                        asyncio.create_subprocess_shell(
                            f"rembg i{' -a' if ctx.parameters.get('alpha') else ''} {temp_file} {temp_file_output}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        ),
                        timeout=15,
                    )
                    stdout, stderr = await terminal.communicate()
                except asyncio.TimeoutError:
                    return await ctx.warn("Couldn't make image **transparent** - Timeout")

                if not os.path.exists(temp_file_output):
                    return await ctx.warn("Couldn't make image **transparent**")

                await ctx.reply(
                    file=discord.File(temp_file_output),
                )

    @commands.command(
        name="legofy",
        usage="(image)",
        example="dscord.com/chnls/999/..png",
        parameters={
            "palette": {
                "converter": str,
                "description": "The LEGO palette to use",
                "default": "solid",
                "choices": [
                    "solid",
                    "transparent",
                    "effects",
                    "mono",
                ],
            },
            "size": {
                "converter": int,
                "description": "The amount of bricks to use",
                "default": None,
                "minimum": 1,
                "maximum": 20,
                "alises": [
                    "scale",
                ],
            },
        },
        aliases=["lego"],
    )
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def legofy(self, ctx: wock.Context, image: wock.ImageFinder = None):
        """Legofy an image"""

        image = image or await wock.ImageFinder.search(ctx)
        if ".gif" in image:
            return await ctx.warn("**GIFs** are not supported")

        async with ctx.typing():
            response = await self.bot.session.get(image)
            if sys.getsizeof(response.content) > 15728640:
                return await ctx.warn("Image is too large to **legofy** (max 15MB)")

            image = await response.read()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(
                    temp_dir, f"file{functions.hash(str(response.url))}." + regex.IMAGE_URL.match(str(response.url)).group("mime")
                )
                async with async_open(temp_file, "wb") as file:
                    await file.write(image)

                try:
                    terminal = await asyncio.wait_for(
                        asyncio.create_subprocess_shell(
                            f"legofy --palette {ctx.parameters.get('palette')} "
                            + (f"--size {ctx.parameters.get('size')} " if ctx.parameters.get("size") else "")
                            + f'"{temp_file}"',
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        ),
                        timeout=10,
                    )
                    stdout, stderr = await terminal.communicate()
                except asyncio.TimeoutError:
                    return await ctx.warn("Couldn't **legofy** image - Timeout")

                file = stdout.decode().split("will now legofy to ")[1].split("\n")[0].strip()

                if not os.path.exists(file):
                    return await ctx.warn("Couldn't **legofy** image")

                await ctx.reply(
                    file=discord.File(file),
                )

    @commands.command(
        name="rotate",
        usage="(image) <degree>",
        example="dscord.com/chnls/999/..png 90",
    )
    @commands.cooldown(1, 6, commands.BucketType.user)
    async def rotate(self, ctx: wock.Context, image: Optional[wock.ImageFinderStrict] = None, degree: int = 90):
        """Rotate an image"""

        image = image or await wock.ImageFinderStrict.search(ctx)

        if degree < 1 or degree > 360:
            return await ctx.warn("Degree must be between **1** and **360**")

        async with ctx.typing():
            response = await self.bot.session.get(image)
            if sys.getsizeof(response.content) > 15728640:
                return await ctx.warn("Image is too large to **rotate** (max 15MB)")

            image = await response.read()

            buffer = await functions.rotate(image, degree)
            await ctx.reply(
                content=f"Rotated **{degree}°** degree" + ("s" if degree != 1 else ""),
                file=discord.File(buffer, filename=f"wockRotate{functions.hash(str(response.url))}.png"),
            )

    @commands.command(
        name="scrapbook",
        usage="(text)",
        example="wock so sexy",
        aliases=["scrap"],
    )
    @commands.max_concurrency(1, commands.BucketType.user)
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def scrapbook(self, ctx: wock.Context, *, text: str):
        """Make scrapbook letters"""

        if len(text) > 20:
            return await ctx.warn("Your text can't be longer than **20 characters**")

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://api.jeyy.xyz/image/scrapbook",
                params=dict(text=text),
            )
            if response.status != 200:
                return await ctx.warn("Couldn't **scrapbook** text - Try again later!")

            image = await response.read()
            buffer = io.BytesIO(image)
            await ctx.reply(
                file=discord.File(
                    buffer,
                    filename=f"wockScrapbook{functions.hash(text)}.gif",
                )
            )

    @commands.command(name="tictactoe", usage="(member)", example="rx#1337", aliases=["ttt"])
    @commands.max_concurrency(1, commands.BucketType.member)
    async def tictactoe(self, ctx: wock.Context, member: wock.Member):
        """Play Tic Tac Toe with another member"""

        if member == ctx.author:
            return await ctx.warn("You can't play against **yourself**")
        elif member.bot:
            return await ctx.warn("You can't play against **bots**")

        await views.TicTacToe(ctx, member).start()

    @commands.command(name="chatgpt", usage="(prompt)", example="I love you..", aliases=["chat", "gpt", "ask", "ai"])
    @commands.max_concurrency(1, commands.BucketType.member)
    @checks.donator()
    async def chatgpt(self, ctx: wock.Context, *, prompt: str):
        """Interact with ChatGPT"""

        await ctx.typing()
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        message = (
            response["choices"][0]["message"]["content"]
            .replace(" As an AI language model, ", "")
            .replace("As an AI language model, ", "")
            .replace(" but as an AI language model, ", "")
        )

        await ctx.reply(message, allowed_mentions=discord.AllowedMentions.none())

    @commands.group(
        name="blunt",
        usage="(subcommand) <args>",
        example="pass rx#1337",
        aliases=["joint"],
        invoke_without_command=True,
        hidden=False,
    )
    async def blunt(self, ctx: wock.Context):
        """Hit the blunt with your homies"""

        await ctx.send_help()

    @blunt.command(
        name="light",
        aliases=["roll"],
        hidden=False,
    )
    async def blunt_light(self, ctx: wock.Context):
        """Roll up a blunt"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if blunt:
            user = ctx.guild.get_member(blunt.get("user_id"))
            return await ctx.warn(
                f"A **blunt** is already held by **{user or blunt.get('user_id')}**\n> It has been hit"
                f" {functions.plural(blunt.get('hits'), bold=True):time} by {functions.plural(blunt.get('members'), bold=True):member}",
            )

        await self.bot.db.execute(
            "INSERT INTO blunt (guild_id, user_id) VALUES($1, $2)",
            ctx.guild.id,
            ctx.author.id,
        )

        await ctx.load("Rolling the **blunt**..", emoji=self.bot.config["styles"]["lighter"].get("emoji"))
        await asyncio.sleep(2)
        await ctx.approve(
            f"Lit up a **blunt**\n> Use `{ctx.prefix}blunt hit` to smoke it",
            emoji="🚬",
        )

    @blunt.command(
        name="pass",
        usage="(member)",
        example="rx#1337",
        aliases=["give"],
        hidden=False,
    )
    async def blunt_pass(self, ctx: wock.Context, *, member: wock.Member):
        """Pass the blunt to another member"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if not blunt:
            return await ctx.warn(f"There is no **blunt** to pass\n> Use `{ctx.prefix}blunt light` to roll one up")
        elif blunt.get("user_id") != ctx.author.id:
            member = ctx.guild.get_member(blunt.get("user_id"))
            return await ctx.warn(f"You don't have the **blunt**!\n> Steal it from **{member or blunt.get('user_id')}** first")
        elif member == ctx.author:
            return await ctx.warn("You can't pass the **blunt** to **yourself**")

        await self.bot.db.execute(
            "UPDATE blunt SET user_id = $2, passes = passes + 1 WHERE guild_id = $1",
            ctx.guild.id,
            member.id,
        )

        await ctx.approve(
            f"The **blunt** has been passed to **{member}**!\n> It has been passed around"
            f" {functions.plural(blunt.get('passes') + 1, bold=True):time}",
            emoji="🚬",
        )

    @blunt.command(
        name="steal",
        aliases=["take"],
        hidden=False,
    )
    @commands.cooldown(1, 60, commands.BucketType.member)
    async def blunt_steal(self, ctx: wock.Context):
        """Steal the blunt from another member"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if not blunt:
            return await ctx.warn(f"There is no **blunt** to steal\n> Use `{ctx.prefix}blunt light` to roll one up")
        elif blunt.get("user_id") == ctx.author.id:
            return await ctx.warn(f"You already have the **blunt**!\n> Use `{ctx.prefix}blunt pass` to pass it to someone else")

        member = ctx.guild.get_member(blunt.get("user_id"))
        if member:
            if member.guild_permissions.manage_messages and not ctx.author.guild_permissions.manage_messages:
                return await ctx.warn(f"You can't steal the **blunt** from **staff** members!")

        # 50% chance that the blunt gets hogged
        if random.randint(1, 100) <= 50:
            return await ctx.warn(f"**{member or blunt.get('user_id')}** is hogging the **blunt**!")

        await self.bot.db.execute(
            "UPDATE blunt SET user_id = $2 WHERE guild_id = $1",
            ctx.guild.id,
            ctx.author.id,
        )

        await ctx.approve(
            f"You just stole the **blunt** from **{member or blunt.get('user_id')}**!",
            emoji="🚬",
        )

    @blunt.command(
        name="hit",
        aliases=["smoke", "chief"],
        hidden=False,
    )
    @commands.max_concurrency(1, commands.BucketType.guild)
    async def blunt_hit(self, ctx: wock.Context):
        """Hit the blunt"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if not blunt:
            return await ctx.warn(f"There is no **blunt** to hit\n> Use `{ctx.prefix}blunt light` to roll one up")
        elif blunt.get("user_id") != ctx.author.id:
            member = ctx.guild.get_member(blunt.get("user_id"))
            return await ctx.warn(f"You don't have the **blunt**!\n> Steal it from **{member or blunt.get('user_id')}** first")

        if not ctx.author.id in blunt.get("members"):
            blunt["members"].append(ctx.author.id)

        await ctx.load(
            "Hitting the **blunt**..",
            emoji="🚬",
        )
        await asyncio.sleep(random.randint(1, 2))

        # 25% chance the blunt burns out
        if blunt["hits"] + 1 >= 10 and random.randint(1, 100) <= 25:
            await self.bot.db.execute(
                "DELETE FROM blunt WHERE guild_id = $1",
                ctx.guild.id,
            )
            return await ctx.warn(
                f"The **blunt** burned out after {functions.plural(blunt.get('hits') + 1, bold=True):hit} by"
                f" {functions.plural(blunt.get('members'), bold=True):member}"
            )

        await self.bot.db.execute(
            "UPDATE blunt SET hits = hits + 1, members = $2 WHERE guild_id = $1",
            ctx.guild.id,
            blunt["members"],
        )

        await ctx.approve(
            f"You just hit the **blunt**!\n> It has been hit {functions.plural(blunt.get('hits') + 1, bold=True):time} by"
            f" {functions.plural(blunt.get('members'), bold=True):member}",
            emoji="🌬",
        )


async def setup(bot: wock.wockSuper):
    await bot.add_cog(fun(bot))
