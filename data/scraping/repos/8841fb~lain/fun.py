import logging
import os
import sys
from asyncio import create_subprocess_shell, sleep, subprocess, wait_for
from contextlib import suppress
from io import BytesIO
from random import choice, randint
from tempfile import TemporaryDirectory
from time import time
from typing import Literal, Union

import openai  # type: ignore
from aiofiles import open as async_open
from discord import (
    AllowedMentions,
    ButtonStyle,
    File,
    SelectOption,  # type: ignore
    HTTPException,
    Interaction,
    Member,
    Message,
)
from discord.ext.commands import BucketType, command, cooldown, group, max_concurrency
from discord.ui import Button, Select, View, Select  # type: ignore

from tools import services
from tools.converters.basic import ImageFinderStrict
from tools.lain import lain
from tools.managers.cog import Cog
from tools.managers.context import Context
from tools.managers.regex import IMAGE_URL
from tools.utilities.checks import donator
from tools.utilities.text import Plural


class Fun(Cog):
    """Cog for Fun"""

    @command(name="8ball", usage="(question)", example="am I pretty?", aliases=["8b"])
    async def eightball(self, ctx: Context, *, question: str):
        """Ask the magic 8ball a question"""

        await ctx.load("Shaking the **magic 8ball**..")

        shakes = randint(1, 5)
        response = choice(list(self.bot.eightball_responses.keys()))
        await getattr(
            ctx, ("approve" if self.bot.eightball_responses[response] else "error")
        )(f"After `{Plural(shakes):shake}` - **{response}**")

    @command(
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
    @donator()
    @cooldown(1, 10, BucketType.user)
    @max_concurrency(1, BucketType.user)
    async def transparent(self, ctx: Context, *, image: ImageFinderStrict = None):
        """Remove the background of an image"""

        image = image or await ImageFinderStrict.search(ctx)

        async with ctx.typing():
            response = await self.bot.session.get(image)
            if sys.getsizeof(response.content) > 15728640:
                return await ctx.error(
                    "Image is too large to make **transparent** (max 15MB)"
                )

            image = await response.read()

            with TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(
                    temp_dir,
                    f"file{hash(str(response.url))}."
                    + IMAGE_URL.match(str(response.url)).group("mime"),
                )
                temp_file_output = os.path.join(
                    temp_dir,
                    f"file{hash(str(response.url))}_output."
                    + IMAGE_URL.match(str(response.url)).group("mime"),
                )
                async with async_open(temp_file, "wb") as file:
                    await file.write(image)

                try:
                    terminal = await wait_for(
                        create_subprocess_shell(
                            f"rembg i{' -a' if ctx.parameters.get('alpha') else ''} {temp_file} {temp_file_output}",
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        ),
                        timeout=15,
                    )
                    stdout, stderr = await terminal.communicate()
                except TimeoutError:
                    return await ctx.error(
                        "Couldn't make image **transparent** - Timeout"
                    )

                if not os.path.exists(temp_file_output):
                    return await ctx.error("Couldn't make image **transparent**")

                await ctx.reply(
                    file=File(temp_file_output),
                )

    @command(
        name="scrapbook",
        usage="(text)",
        example="lain so sexy",
        aliases=["scrap"],
    )
    @max_concurrency(1, BucketType.user)
    @cooldown(1, 10, BucketType.user)
    async def scrapbook(self: "Fun", ctx: Context, *, text: str):
        """Make scrapbook letters"""

        if len(text) > 20:
            return await ctx.error("Your text can't be longer than **20 characters**")

        async with ctx.typing():
            response = await self.bot.session.request(
                "GET",
                "https://api.jeyy.xyz/image/scrapbook",
                params=dict(text=text),
            )

            buffer = BytesIO(response)
            await ctx.reply(
                file=File(
                    buffer,
                    filename=f"lainScrapbook.gif",
                )
            )

    @command(name="roll", usage="(sides)", example="6", aliases=["dice"])
    async def roll(self: "Fun", ctx: Context, sides: int = 6):
        """Roll a dice"""

        await ctx.load(f"Rolling a **{sides}-sided** dice..")

        await ctx.approve(f"The dice landed on **🎲 {randint(1, sides)}**")

    @command(
        name="coinflip",
        usage="<heads/tails>",
        example="heads",
        aliases=["flipcoin", "cf", "fc"],
    )
    async def coinflip(
        self: "Fun", ctx: Context, *, side: Literal["heads", "tails"] = None
    ):
        """Flip a coin"""

        await ctx.load(
            f"Flipping a coin{f' and guessing **:coin: {side}**' if side else ''}.."
        )

        coin = choice(["heads", "tails"])
        await getattr(ctx, ("approve" if (not side or side == coin) else "error"))(
            f"The coin landed on **:coin: {coin}**"
            + (f", you **{'won' if side == coin else 'lost'}**!" if side else "!")
        )

    @command(name="tictactoe", usage="(member)", example="caden", aliases=["ttt"])
    @max_concurrency(1, BucketType.member)
    async def tictactoe(self: "Fun", ctx: Context, member: Member):
        """Play Tic Tac Toe with another member"""

        if member == ctx.author:
            return await ctx.error("You can't play against **yourself**")
        elif member.bot:
            return await ctx.error("You can't play against **bots**")

        await services.TicTacToe(ctx, member).start()

    @command(
        name="chatgpt",
        usage="(prompt)",
        example="I love you..",
        aliases=["chat", "gpt", "ask", "ai"],
    )
    @max_concurrency(1, BucketType.member)
    @donator()
    async def chatgpt(self: "Fun", ctx: Context, *, prompt: str):
        """Interact with ChatGPT"""

        await ctx.typing()
        _start = time()
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

        logging.info(
            f"Obtained ChatGPT response ({time() - _start:.2f}s) - {ctx.author} ({ctx.author.id})"
        )

        message = (
            response["choices"][0]["message"]["content"]
            .replace(" As an AI language model, ", "")
            .replace("As an AI language model, ", "")
            .replace(" but as an AI language model, ", "")
        )

        await ctx.reply(message, allowed_mentions=AllowedMentions.none())

    @command(
        name="marry",
        usage="(member)",
        example="caden",
        aliases=["propose", "proposal"],
    )
    @max_concurrency(1, BucketType.member)
    @cooldown(1, 60, BucketType.member)
    async def marry(self: "Fun", ctx: Context, member: Member):
        """Propose to another member"""

        marriage = await self.bot.db.fetchrow(
            "SELECT * FROM marriages WHERE user_id = $1 OR partner_id = $1",
            member.id,
        )
        if marriage:
            return await ctx.error(
                f"**{member.name}** is already married to **{self.bot.get_user(marriage.get('user_id')).name}**"
            )

        marriage = await self.bot.db.fetchrow(
            "SELECT * FROM marriages WHERE user_id = $1 OR partner_id = $1",
            ctx.author.id,
        )
        if marriage:
            return await ctx.error(
                f"You're already married to **{self.bot.get_user(marriage.get('user_id')).name}**"
            )

        if member == ctx.author:
            return await ctx.error("You can't marry **yourself**")

        if member.bot:
            return await ctx.error("You can't marry **bots**")

        if not await ctx.prompt(
            f"**{member.name}**, do you accept **{ctx.author.name}**'s proposal?",
            member=member,
        ):
            return await ctx.error(f"**{member.name}** denied your proposal")

        await self.bot.db.execute(
            "INSERT INTO marriages (user_id, partner_id) VALUES ($1, $2)",
            ctx.author.id,
            member.id,
        )

        return await ctx.neutral(
            f"**{ctx.author.name}** and **{member.name}** are now married!"
        )

    @command(
        name="divorce",
        aliases=["breakup"],
    )
    @max_concurrency(1, BucketType.member)
    @cooldown(1, 60, BucketType.member)
    async def divorce(self: "Fun", ctx: Context):
        """Divorce your partner"""

        marriage = await self.bot.db.fetchrow(
            "SELECT * FROM marriages WHERE user_id = $1 OR partner_id = $1",
            ctx.author.id,
        )
        if not marriage:
            return await ctx.error("You're not **married** to anyone")

        await ctx.prompt(
            f"Are you sure you want to divorce **{self.bot.get_user(marriage.get('partner_id')).name}**?",
        )

        await self.bot.db.execute(
            "DELETE FROM marriages WHERE user_id = $1 OR partner_id = $1", ctx.author.id
        )
        return await ctx.neutral("You are now **divorced**")

    @command(
        name="partner",
        aliases=["spouse"],
    )
    @max_concurrency(1, BucketType.member)
    async def partner(self: "Fun", ctx: Context):
        """Check who you're married to"""

        marriage = await self.bot.db.fetchrow(
            "SELECT * FROM marriages WHERE user_id = $1 OR partner_id = $1",
            ctx.author.id,
        )
        if not marriage:
            return await ctx.error("You're not **married** to anyone")

        partner = self.bot.get_user(marriage.get("partner_id"))
        return await ctx.neutral(f"You're married to **{partner}**")

    @group(
        name="blunt",
        usage="(subcommand) <args>",
        example="pass caden",
        aliases=["joint"],
        invoke_without_command=True,
        hidden=False,
    )
    async def blunt(self: "Fun", ctx: Context):
        """Hit the blunt with your homies"""

        await ctx.send_help()

    @blunt.command(
        name="light",
        aliases=["roll"],
        hidden=False,
    )
    async def blunt_light(self: "Fun", ctx: Context):
        """Roll up a blunt"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if blunt:
            user = ctx.guild.get_member(blunt.get("user_id"))
            return await ctx.error(
                f"A **blunt** is already held by **{user or blunt.get('user_id')}**\n> It has been hit"
                f" {Plural(blunt.get('hits')):time} by {Plural(blunt.get('members')):member}",
            )

        await self.bot.db.execute(
            "INSERT INTO blunt (guild_id, user_id) VALUES($1, $2)",
            ctx.guild.id,
            ctx.author.id,
        )

        await ctx.load(
            "Rolling the **blunt**..", emoji="<:lighter:1135645885091553330>"
        )
        await sleep(2)
        await ctx.approve(
            f"Lit up a **blunt**\n> Use `{ctx.prefix}blunt hit` to smoke it",
            emoji="🚬",
        )

    @blunt.command(
        name="pass",
        usage="(member)",
        example="caden",
        aliases=["give"],
        hidden=False,
    )
    async def blunt_pass(self: "Fun", ctx: Context, *, member: Member):
        """Pass the blunt to another member"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if not blunt:
            return await ctx.error(
                f"There is no **blunt** to pass\n> Use `{ctx.prefix}blunt light` to roll one up"
            )
        elif blunt.get("user_id") != ctx.author.id:
            member = ctx.guild.get_member(blunt.get("user_id"))
            return await ctx.error(
                f"You don't have the **blunt**!\n> Steal it from **{member or blunt.get('user_id')}** first"
            )
        elif member == ctx.author:
            return await ctx.error("You can't pass the **blunt** to **yourself**")

        await self.bot.db.execute(
            "UPDATE blunt SET user_id = $2, passes = passes + 1 WHERE guild_id = $1",
            ctx.guild.id,
            member.id,
        )

        await ctx.approve(
            f"The **blunt** has been passed to **{member}**!\n> It has been passed around"
            f" **{Plural(blunt.get('passes') + 1):time}**",
            emoji="🚬",
        )

    @blunt.command(
        name="steal",
        aliases=["take"],
        hidden=False,
    )
    @cooldown(1, 60, BucketType.member)
    async def blunt_steal(self: "Fun", ctx: Context):
        """Steal the blunt from another member"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if not blunt:
            return await ctx.error(
                f"There is no **blunt** to steal\n> Use `{ctx.prefix}blunt light` to roll one up"
            )
        elif blunt.get("user_id") == ctx.author.id:
            return await ctx.error(
                f"You already have the **blunt**!\n> Use `{ctx.prefix}blunt pass` to pass it to someone else"
            )

        member = ctx.guild.get_member(blunt.get("user_id"))
        if member:
            if (
                member.guild_permissions.manage_messages
                and not ctx.author.guild_permissions.manage_messages
            ):
                return await ctx.error(
                    f"You can't steal the **blunt** from **staff** members!"
                )

        # 50% chance that the blunt gets hogged
        if randint(1, 100) <= 50:
            return await ctx.error(
                f"**{member or blunt.get('user_id')}** is hogging the **blunt**!"
            )

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
    @max_concurrency(1, BucketType.guild)
    async def blunt_hit(self: "Fun", ctx: Context):
        """Hit the blunt"""

        blunt = await self.bot.db.fetchrow(
            "SELECT * FROM blunt WHERE guild_id = $1",
            ctx.guild.id,
        )
        if not blunt:
            return await ctx.error(
                f"There is no **blunt** to hit\n> Use `{ctx.prefix}blunt light` to roll one up"
            )
        elif blunt.get("user_id") != ctx.author.id:
            member = ctx.guild.get_member(blunt.get("user_id"))
            return await ctx.error(
                f"You don't have the **blunt**!\n> Steal it from **{member or blunt.get('user_id')}** first"
            )

        if not ctx.author.id in blunt.get("members"):
            blunt["members"].append(ctx.author.id)

        await ctx.load(
            "Hitting the **blunt**..",
            emoji="🚬",
        )
        await sleep(randint(1, 2))

        # 25% chance the blunt burns out
        if blunt["hits"] + 1 >= 10 and randint(1, 100) <= 25:
            await self.bot.db.execute(
                "DELETE FROM blunt WHERE guild_id = $1",
                ctx.guild.id,
            )
            return await ctx.error(
                f"The **blunt** burned out after {Plural(blunt.get('hits') + 1):hit} by"
                f" **{Plural(blunt.get('members')):member}**"
            )

        await self.bot.db.execute(
            "UPDATE blunt SET hits = hits + 1, members = $2 WHERE guild_id = $1",
            ctx.guild.id,
            blunt["members"],
        )

        await ctx.approve(
            f"You just hit the **blunt**!\n> It has been hit **{Plural(blunt.get('hits') + 1):time}** by"
            f" **{Plural(blunt.get('members')):member}**",
            emoji="🌬",
        )

    @command(name="spam", aliases=["funnycaincommand"])
    async def spam(self: "Fun", ctx: Context, amount: int = 5, *, text: str):
        """funny cain cmd for cain only haha Losers"""

        if ctx.author.id != 1113879430767575150:
            return await ctx.message.add_reaction("❌")

        # Limit the amount of spam
        if amount > 15:
            return await ctx.message.add_reaction("❌")

        # Limit the amount of characters
        if len(text) > 1500:
            return await ctx.message.add_reaction("❌")

        for _ in range(amount):
            await ctx.send(text)

        await ctx.message.add_reaction("✅")
        await ctx.message.add_reaction("✨")

    @command(name="wizz", aliases=["hahafunnycaincomamndnumber2"])
    async def wizz(self: "Fun", ctx: Context):
        """funny cain cmd for cain only haha Losers (no this doesnt work idiot)"""

        if ctx.author.id != 1113879430767575150:
            return await ctx.message.add_reaction("❌")

        await ctx.load("Wizzing the server..")
        await ctx.load("Deleting all roles..")
        await ctx.load("Deleting all channels..")
        await ctx.load("Banning all members..")
        await ctx.load("Deleting all emojis..")
        await ctx.load("Deleting all webhooks..")
        await ctx.load("Deleting all stickers..")
        await ctx.approve(
            "**GET FUCKING WIZZED BY FUCKING CAIN LOSERS LAOOLSAI FHJUHAIJS!!!!!**"
        )

        await ctx.message.add_reaction("✅")
        await ctx.message.add_reaction("✨")
