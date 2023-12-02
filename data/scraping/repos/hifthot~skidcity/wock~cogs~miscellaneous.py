import asyncio
import base64
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time

from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

import discord
import openai
import yarl

from aiofiles import open as async_open
from asyncspotify import Client as SpotifyClient
from asyncspotify import ClientCredentialsFlow as SpotifyClientCredentialsFlow
from discord.ext import commands, tasks

from helpers import checks, functions, humanize, models, regex, rtfm, tuuid, views, wock


class miscellaneous(commands.Cog, name="Miscellaneous"):
    def __init__(self, bot):
        self.bot: wock.wockSuper = bot
        self.buckets: dict = dict(
            emoji=dict(lock=asyncio.Lock(), data=defaultdict(Counter)),
            avatars=dict(
                lock=asyncio.Lock(),
                data=defaultdict(dict),
            ),
            seen=dict(
                lock=asyncio.Lock(),
                data=defaultdict(dict),
            ),
        )
        self.emoji_insert.start()
        self.seen_insert.start()
        self.btc_notifier.start()
        self.reminder.start()
        self.spotify_client = SpotifyClient(
            SpotifyClientCredentialsFlow(
                client_id=self.bot.config["api"]["spotify"]["client_id"],
                client_secret=self.bot.config["api"]["spotify"]["client_secret"],
            )
        )
        self.bot.loop.create_task(self.spotify_client.authorize())
        openai.api_key = self.bot.config["api"]["openai"]

    def cog_unload(self):
        self.emoji_insert.stop()
        self.seen_insert.stop()
        self.btc_notifier.stop()
        self.reminder.stop()

    async def sport_scores(self, sport: str):
        """Generate the embeds for the scores of a sport"""

        response = await self.bot.session.get(f"http://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard")
        data = await response.json()

        if not data.get("events"):
            raise commands.CommandError(f"There aren't any **{sport.split('/')[0].title()}** events!")

        embeds = []
        for event in data["events"]:
            embed = discord.Embed(url=f"https://www.espn.com/{sport.split('/')[1]}/game?gameId={event['id']}", title=event.get("name"))
            embed.set_author(
                name=event["competitions"][0]["competitors"][0]["team"]["displayName"],
                icon_url=event["competitions"][0]["competitors"][0]["team"]["logo"],
            )
            embed.set_thumbnail(
                url=event["competitions"][0]["competitors"][1]["team"]["logo"],
            )
            embed.add_field(
                name="Status",
                value=event["status"]["type"].get("detail"),
                inline=True,
            )
            embed.add_field(
                name="Teams",
                value=(
                    f"{event['competitions'][0]['competitors'][1]['team']['abbreviation']} -"
                    f" {event['competitions'][0]['competitors'][0]['team']['abbreviation']}"
                ),
                inline=True,
            )
            embed.add_field(
                name="Score",
                value=f"{event['competitions'][0]['competitors'][1]['score']} - {event['competitions'][0]['competitors'][0]['score']}",
                inline=True,
            )
            embed.timestamp
            embeds.append(embed)

        return embeds

    @tasks.loop(seconds=60)
    async def emoji_insert(self):
        """Bulk insert emojis saved in the bucket into the database"""

        bucket = self.buckets.get("emoji")
        async with bucket["lock"]:
            transformed = [
                dict(
                    guild_id=int(guild_id),
                    emoji_id=int(emoji_id),
                    uses=count,
                )
                for guild_id, data in bucket["data"].items()
                for emoji_id, count in data.items()
            ]
            bucket["data"].clear()

            await self.bot.db.execute(
                "INSERT INTO metrics.emojis (guild_id, emoji_id, uses, timestamp) SELECT x.guild_id, x.emoji_id, x.uses, $2 FROM"
                " jsonb_to_recordset($1::JSONB) AS x(guild_id BIGINT, emoji_id BIGINT, uses BIGINT) ON CONFLICT (guild_id, emoji_id) DO UPDATE SET"
                " uses = metrics.emojis.uses + EXCLUDED.uses",
                transformed,
                discord.utils.utcnow(),
            )

    @emoji_insert.before_loop
    async def emoji_insert_before(self):
        await self.bot.wait_until_ready()

    @commands.Cog.listener("on_user_update")
    async def avatar_update(self, before: discord.User, after: discord.User):
        """Save past avatars to the upload bucket"""

        if not self.bot.is_ready() or not after.avatar or str(before.display_avatar) == str(after.display_avatar):
            return

        channel = self.bot.get_channel(self.bot.config["channels"]["avatars"])
        if not channel:
            return

        try:
            image = await after.avatar.read()
        except:
            return  # asset too new

        image_hash = await functions.image_hash(image)

        with contextlib.suppress(discord.HTTPException):
            message = await channel.send(
                file=discord.File(
                    io.BytesIO(image),
                    filename=f"{image_hash}." + ("png" if not before.display_avatar.is_animated() else "gif"),
                )
            )

            await self.bot.db.execute(
                "INSERT INTO metrics.avatars (user_id, avatar, hash, timestamp) VALUES ($1, $2, $3, $4) ON CONFLICT (user_id, hash) DO NOTHING",
                before.id,
                message.attachments[0].url,
                image_hash,
                int(discord.utils.utcnow().timestamp()),
            )
            # self.bot.logger.info(f"Saved asset {image_hash} for {before}")

    @tasks.loop(seconds=60)
    async def seen_insert(self):
        """Bulk insert seen data saved in the bucket into the database"""

        bucket = self.buckets.get("seen")
        async with bucket["lock"]:
            transformed = [
                dict(
                    user_id=int(user_id),
                    timestamp=data,
                )
                for user_id, data in bucket["data"].items()
            ]
            bucket["data"].clear()

            await self.bot.db.execute(
                "INSERT INTO metrics.seen (user_id, timestamp) SELECT x.user_id, x.timestamp FROM"
                " jsonb_to_recordset($1::JSONB) AS x(user_id BIGINT, timestamp BIGINT) ON CONFLICT (user_id) DO UPDATE SET"
                " timestamp = EXCLUDED.timestamp",
                transformed,
            )

    @seen_insert.before_loop
    async def seen_insert_before(self):
        await self.bot.wait_until_ready()

    @commands.Cog.listener("on_user_activity")
    async def seen_update(self, channel: discord.TextChannel, member: discord.Member):
        """Save when a user was last seen in a channel"""

        bucket = self.buckets.get("seen")
        async with bucket["lock"]:
            bucket["data"][member.id] = int(discord.utils.utcnow().timestamp())

    @tasks.loop(seconds=60)
    async def btc_notifier(self):
        """Notify when a transaction receives a confirmation"""

        async for subscription in self.bot.db.fetchiter("SELECT * FROM btc_subscriptions"):
            if user := self.bot.get_user(subscription.get("user_id")):
                response = await self.bot.session.get("https://mempool.space/api/tx/" + subscription.get("transaction") + "/status")
                if response.status != 200:
                    await self.bot.db.execute(
                        "DELETE FROM btc_subscriptions WHERE user_id = $1 AND transaction = $2",
                        subscription.get("user_id"),
                        subscription.get("transaction"),
                    )
                    continue

                data = await response.json()
                if data.get("confirmed"):
                    with contextlib.suppress(discord.HTTPException):
                        await user.send(
                            embed=discord.Embed(
                                color=functions.config_color("main"),
                                description=(
                                    "💸 Your **transaction**"
                                    f" [`{functions.shorten(subscription.get('transaction'))}`](https://mempool.space/tx/{subscription.get('transaction')})"
                                    " has received a **confirmation**!"
                                ),
                            )
                        )

                    await self.bot.db.execute(
                        "DELETE FROM btc_subscriptions WHERE user_id = $1 AND transaction = $2",
                        subscription.get("user_id"),
                        subscription.get("transaction"),
                    )

    @btc_notifier.before_loop
    async def btc_notifier_before(self):
        await self.bot.wait_until_ready()

    # @tasks.loop(seconds=60)
    # async def tiktok_feed(self):
    #     """Notify channels whenever a TikTok profile posts"""

    #     async for feed in self.bot.db.fetchiter("SELECT * FROM tiktok"):
    #         if (channels := [self.bot.get_channel(channel_id) for channel_id in feed["channel_ids"]]):
    #             username = feed["username"]
    #             post_id = feed["post_id"]

    #             response = await self.bot.session.get(
    #                 "https://dev.wock.cloud/tiktok/profile",
    #                 params=dict(username=username),
    #                 headers=dict(Authorization=self.bot.config["api"].get("wock")),
    #             )
    #             data = await response.json()

    #             if (error := data.get("error")) and error == "Invalid TikTok username.":
    #                 await self.bot.db.execute("DELETE FROM tiktok WHERE username = $1", username)
    #                 continue
    #             elif not data.get("videos"):
    #                 continue
    #             else:
    #                 account = models.TikTokUser(**data)

    #             if account.videos[0].id == post_id:
    #                 continue
    #             else:
    #                 await self.bot.db.execute("UPDATE tiktok SET post_id = $2 WHERE username = $1", username, account.videos[0].id)

    #             for channel in channels:
    #                 await functions.ensure_future(
    #                     channel.send(
    #                         content=f"Yo.. **{account.nickname}** posted <:grinnin:1094116095310442526>\n{account.videos[0].url}"
    #                     )
    #                 )

    # @tiktok_feed.before_loop
    # async def tiktok_feed_before(self):
    #     await self.bot.wait_until_ready()

    @tasks.loop(seconds=30)
    async def reminder(self):
        """Notify when a reminder is due"""

        async for reminder in self.bot.db.fetchiter("SELECT * FROM reminders"):
            if user := self.bot.get_user(reminder.get("user_id")):
                if discord.utils.utcnow() >= reminder.get("timestamp"):
                    with contextlib.suppress(discord.HTTPException):
                        await user.send(
                            embed=discord.Embed(
                                color=functions.config_color("main"),
                                description=(
                                    f"⏰ You wanted me to remind you to **{reminder.get('text')}**"
                                    f" ({discord.utils.format_dt(reminder.get('created_at'), style='f')})"
                                ),
                            ),
                            view=views.Reminder(reminder.get("jump_url")),
                        )
                    await self.bot.db.execute("DELETE FROM reminders WHERE user_id = $1 AND text = $2", reminder.get("user_id"), reminder.get("text"))

    @commands.Cog.listener("on_user_update")
    async def username_update(self, before: discord.User, after: discord.User):
        """Save past names to the database"""

        if not self.bot.is_ready() or before.name == after.name:
            return

        await self.bot.db.execute(
            "INSERT INTO metrics.names (user_id, name, timestamp) VALUES ($1, $2, $3)",
            after.id,
            str(before),
            discord.utils.utcnow(),
        )

    @commands.Cog.listener("on_user_message")
    async def message_repost(self, ctx: wock.Context, message: discord.Message):
        """Repost message links"""

        if not message.content:
            return
        if not "discord.com/channels" in message.content:
            return

        if match := regex.DISCORD_MESSAGE.match(message.content):
            guild_id, channel_id, message_id = map(int, match.groups())
            if guild_id != ctx.guild.id:
                return
            channel = self.bot.get_channel(channel_id)
            if not channel:
                return
            if not channel.permissions_for(ctx.me).view_channel:
                return
            if not channel.permissions_for(ctx.author).view_channel:
                return
        else:
            return

        bucket = self.bot.buckets.get("message_reposting").get_bucket(ctx.message)
        if bucket.update_rate_limit():
            return

        try:
            message = await channel.fetch_message(message_id)
        except discord.HTTPException:
            return

        if message.embeds and not message.embeds[0].type == "image":
            embed = message.embeds[0]
            embed.description = embed.description or ""
        else:
            embed = discord.Embed(
                color=(message.author.color if message.author.color.value else functions.config_color("main")),
                description="",
            )
        embed.set_author(name=message.author, icon_url=message.author.display_avatar, url=message.jump_url)

        if message.content:
            embed.description += f"\n{message.content}"

        if message.attachments and message.attachments[0].content_type.startswith("image"):
            embed.set_image(url=message.attachments[0].proxy_url)

        attachments = list()
        for attachment in message.attachments:
            if attachment.content_type.startswith("image"):
                continue
            if attachment.size > ctx.guild.filesize_limit:
                continue
            if not attachment.filename.endswith(("mp4", "mp3", "mov", "wav", "ogg", "webm")):
                continue

            attachments.append(await attachment.to_file())

        embed.set_footer(text=f"Posted @ #{message.channel}", icon_url=message.guild.icon)
        embed.timestamp = message.created_at

        await ctx.channel.send(embed=embed, files=attachments, reference=(ctx.replied_message or ctx.message))

    @commands.Cog.listener("on_message_repost")
    async def tiktok_repost(self, ctx: wock.Context, argument: str):
        """Repost TikTok posts"""

        if not "tiktok" in argument:
            return
        if match := (regex.TIKTOK_MOBILE_URL.match(argument) or regex.TIKTOK_DESKTOP_URL.match(argument)):
            argument = match.group()
        else:
            return

        # bucket = self.bot.buckets.get("tiktok_reposting").get_bucket(ctx.message)
        # if retry_after := bucket.update_rate_limit():
        #     return await ctx.warn(
        #         f"Please wait **{retry_after:.2f} seconds** before attempting to repost again",
        #         delete_after=retry_after,
        #     )

        _start = time.time()
        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/tiktok/post",
                params=dict(
                    content=argument,
                    user_id=ctx.author.id,
                    guild_id=ctx.guild.id,
                ),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn("Invalid **TikTok URL**")
        else:
            post = models.TikTokPost(**data)
            self.bot.logger.success(f"Obtained page {post.id} ({time.time() - _start:.2f}s) - {ctx.author} ({ctx.author.id})")

        embed = discord.Embed(
            url=post.share_url,
            title=post.caption.split("\n")[0] if post.caption else None,
        )
        embed.set_author(
            name=post.user.nickname,
            url=post.user.url,
            icon_url=post.user.avatar,
        )

        embed.set_footer(
            text=f"❤️ {post.statistics.likes:,} 💬 {post.statistics.comments:,} 🎬 {post.statistics.plays:,} - {ctx.message.author}",
            icon_url="https://seeklogo.com/images/T/tiktok-icon-logo-1CB398A1BD-seeklogo.com.png",
        )
        embed.timestamp = post.created_at

        if images := post.assets.images:
            embeds = [(embed.copy().set_image(url=image)) for image in images]
            return await ctx.paginate(embeds)

        response = await self.bot.session.get(yarl.URL(post.assets.video, encoded=True))
        file = discord.File(
            io.BytesIO(await response.read()),
            filename=f"wockTikTok{tuuid.random()}.mp4",
        )
        if sys.getsizeof(file.fp) > ctx.guild.filesize_limit:
            return await ctx.warn("The **video** is too large to be sent")

        await ctx.send(embed=embed, file=file)

    @commands.Cog.listener("on_message_repost")
    async def twitter_repost(self, ctx: wock.Context, argument: str):
        """Repost Twitter posts"""

        if not "twitter" in argument:
            return
        if match := regex.TWITTER_URL.match(argument):
            argument = match.group()
        else:
            return

        bucket = self.bot.buckets.get("twitter_reposting").get_bucket(ctx.message)
        if retry_after := bucket.update_rate_limit():
            return await ctx.warn(
                f"Please wait **{retry_after:.2f} seconds** before attempting to repost again",
                delete_after=retry_after,
            )

        _start = time.time()
        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/twitter/post",
                params=dict(
                    content=argument,
                    user_id=ctx.author.id,
                    guild_id=ctx.guild.id,
                ),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn("Invalid **Twitter URL**")
        else:
            post = models.TwitterPost(**data)
            self.bot.logger.success(f"Obtained page {post.id} ({time.time() - _start:.2f}s) - {ctx.author} ({ctx.author.id})")

        embed = discord.Embed(
            url=post.url,
            description=post.text,
        )
        embed.set_author(
            name=post.user.name,
            url=post.user.url,
            icon_url=post.user.avatar,
        )

        embed.set_footer(
            text=f"❤️ {post.statistics.likes:,} 💬 {post.statistics.replies:,} 🔁 {post.statistics.retweets} - {ctx.message.author}",
            icon_url="https://discord.com/assets/4662875160dc4c56954003ebda995414.png",
        )
        embed.timestamp = post.created_at

        if images := post.assets.images:
            embeds = [(embed.copy().set_image(url=image)) for image in images]
            return await ctx.paginate(embeds)

        response = await self.bot.session.get(yarl.URL(post.assets.video, encoded=True))
        file = discord.File(
            io.BytesIO(await response.read()),
            filename=f"wockTwitter{tuuid.random()}.mp4",
        )
        if sys.getsizeof(file.fp) > ctx.guild.filesize_limit:
            return await ctx.warn("The **video** is too large to be sent")

        await ctx.send(embed=embed, file=file)

    # @commands.Cog.listener("on_message_repost")
    # async def instagram_repost(self, ctx: wock.Context, argument: str):
    #     """Repost Instagram posts"""

    #     if not "instagram" in argument:
    #         return
    #     if match := regex.INSTAGRAM_URL.match(argument):
    #         argument = match.group()
    #     else:
    #         return

    #     bucket = self.bot.buckets.get("instagram_reposting").get_bucket(ctx.message)
    #     if retry_after := bucket.update_rate_limit():
    #         return await ctx.warn(
    #             f"Please wait **{retry_after:.2f} seconds** before attempting to repost again",
    #             delete_after=retry_after,
    #         )

    #     _start = time.time()
    #     async with ctx.typing():
    #         response = await self.bot.session.get(
    #             "https://dev.wock.cloud/instagram/post",
    #             params=dict(
    #                 content=argument,
    #                 user_id=ctx.author.id,
    #                 guild_id=ctx.guild.id,
    #                 identifier=identifier,
    #             ),
    #             headers=dict(
    #                 Authorization=self.bot.config["api"].get("wock"),
    #             ),
    #         )
    #         data = await response.json()

    #     if "error" in data:
    #         if "Invalid identifier" in data.get("error"):
    #             return await ctx.warn("Invalid **identifier**")
    #         else:
    #             return await ctx.warn("Invalid **Instagram URL**")
    #     else:
    #         post = models.InstagramPost(**data)
    #         self.bot.logger.success(f"Obtained page {post.shortcode} ({time.time() - _start:.2f}s) - {ctx.author} ({ctx.author.id})")

    #     embed = discord.Embed(
    #         url=post.share_url,
    #         title=post.caption.split("\n")[0] if post.caption else None,
    #     )
    #     embed.set_author(
    #         name=post.user.name,
    #         url=post.user.url,
    #         icon_url=post.user.avatar,
    #     )

    #     embed.set_footer(
    #         text=f"❤️ {post.statistics.likes:,} 💬 {post.statistics.comments:,} - {ctx.message.author}"
    #         + (f" ({identifier} of {post.results})" if post.results > 1 else ""),
    #         icon_url="https://media.discordapp.net/attachments/1015399830685749299/1028640164928569384/68d99ba29cc8_1.png",
    #     )
    #     embed.timestamp = post.created_at

    #     if post.media.type == "image":
    #         embed.set_image(url=post.media.url)
    #         return await ctx.send(embed=embed)

    #     response = await self.bot.session.get(yarl.URL(post.media.url, encoded=True))
    #     file = discord.File(
    #         io.BytesIO(await response.read()),
    #         filename=f"wockIG{tuuid.random()}.mp4",
    #     )
    #     if sys.getsizeof(file.fp) > ctx.guild.filesize_limit:
    #         return await ctx.warn("The **video** is too large to be sent")
    #     elif sys.getsizeof(file.fp) == 22:
    #         return await ctx.reply("I'm so sorry but instagram didn't let me view this video :/")

    #     await ctx.send(embed=embed, file=file)

    @commands.Cog.listener("on_message_repost")
    async def pinterest_repost(self, ctx: wock.Context, argument: str):
        """Repost Pinterest pins"""

        if not "pin" in argument:
            return
        if match := (regex.PINTEREST_PIN_URL.match(argument) or regex.PINTEREST_PIN_APP_URL.match(argument)):
            argument = match.group()
        else:
            return

        _start = time.time()
        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/pinterest/pin",
                params=dict(
                    content=argument,
                ),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn("Invalid **Pinterest URL**")
        else:
            pin = models.PinterestPin(**data)
            self.bot.logger.success(f"Obtained page {pin.id} ({time.time() - _start:.2f}s) - {ctx.author} ({ctx.author.id})")

        embed = discord.Embed(
            url=pin.url,
            title=pin.title,
        )
        embed.set_author(
            name=pin.user.display_name,
            url=pin.user.url,
            icon_url=pin.user.avatar,
        )

        embed.set_footer(
            text=f"❤️ {pin.statistics.saves:,} 💬 {pin.statistics.comments:,} - {ctx.message.author}",
            icon_url="https://cdn-icons-png.flaticon.com/512/174/174863.png",
        )

        if pin.media.type == "image":
            embed.set_image(url=pin.media.url)
            return await ctx.send(embed=embed)

        response = await self.bot.session.get(yarl.URL(pin.media.url, encoded=True))
        file = discord.File(
            io.BytesIO(await response.read()),
            filename=f"wockPinterest{tuuid.random()}.mp4",
        )
        if sys.getsizeof(file.fp) > ctx.guild.filesize_limit:
            return await ctx.warn("The **video** is too large to be sent")

        await ctx.send(embed=embed, file=file)

    @commands.Cog.listener("on_message_repost")
    async def grailed_repost(self, ctx: wock.Context, argument: str):
        """Repost Grailed listings"""

        if not "grailed" in argument:
            return
        if match := (regex.GRAILED_LISTING_URL.match(argument) or regex.GRAILED_LISTING_APP_URL.match(argument)):
            argument = match.group()
        else:
            return

        _start = time.time()
        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/grailed/listing",
                params=dict(
                    content=argument,
                ),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn("Invalid **Grailed URL**")
        else:
            listing = models.GrailedListing(**data)
            self.bot.logger.success(f"Obtained page {listing.id} ({time.time() - _start:.2f}s) - {ctx.author} ({ctx.author.id})")

        embed = discord.Embed(
            url=listing.url,
            title=listing.title,
        )
        embed.set_author(
            name=listing.seller.username,
            url=listing.seller.url,
            icon_url=listing.seller.avatar,
        )

        embed.set_footer(
            text=f"👕 {listing.size.title()} 💸 ${listing.price} {listing.currency}  - {ctx.message.author}",
            icon_url="https://techround.co.uk/wp-content/uploads/2020/02/grailed.png",
        )
        embed.timestamp = listing.created_at

        await ctx.paginate([embed.copy().set_image(url=image) for image in listing.images])

    @commands.Cog.listener("on_message_repost")
    async def youtube_repost(self, ctx: wock.Context, argument: str):
        """Repost YouTube videos"""

        if not "youtu" in argument:
            return
        if match := (regex.YOUTUBE_URL.match(argument) or regex.YOUTUBE_SHORT_URL.match(argument) or regex.YOUTUBE_SHORTS_URL.match(argument)):
            argument = match.group()
        else:
            return

        bucket = self.bot.buckets.get("youtube_reposting").get_bucket(ctx.message)
        if retry_after := bucket.update_rate_limit():
            return await ctx.warn(
                f"Please wait **{retry_after:.2f} seconds** before attempting to repost again",
                delete_after=retry_after,
            )

        _start = time.time()
        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/youtube/video",
                params=dict(
                    content=argument,
                    user_id=ctx.author.id,
                    guild_id=ctx.guild.id,
                ),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn("Invalid **YouTube URL**")
        elif data["download"]["duration"] > 360:
            return await ctx.warn("The **video** is too long to be reposted (`max 6 minutes`)")
        else:
            self.bot.logger.success(f"Obtained page {match.group(1)} ({time.time() - _start:.2f}s) - {ctx.author} ({ctx.author.id})")

        embed = discord.Embed(
            url=data.get("url"),
            title=data.get("title"),
        )
        embed.set_author(
            name=data["user"].get("name"),
            url=data["user"].get("url"),
            icon_url=ctx.author.display_avatar,
        )

        response = await self.bot.session.get(yarl.URL(data["download"].get("url"), encoded=True))
        file = discord.File(
            io.BytesIO(await response.read()),
            filename=f"wockYT{tuuid.random()}.mp4",
        )
        if sys.getsizeof(file.fp) > ctx.guild.filesize_limit:
            return await ctx.warn("The **video** is too large to be sent")

        embed.set_footer(
            text=f"👁‍🗨 {data['statistics'].get('views'):,} - {ctx.message.author}",
            icon_url="https://discord.com/assets/449cca50c1452b4ace3cbe9bc5ae0fd6.png",
        )
        embed.timestamp = datetime.fromtimestamp(data.get("created_at"))
        await ctx.send(embed=embed, file=file)

    @commands.Cog.listener("on_user_message")
    async def repair_videos(self, ctx: wock.Context, message: discord.Message):
        """Repair broken MOV videos"""

        if not message.attachments:
            return

        attachment = message.attachments[0]
        if attachment.content_type in ("video/quicktime", "x-matroska", "video/x-ms-wmv") and not attachment.height:
            bucket = self.bot.buckets.get("highlights").get_bucket(message)
            if bucket.update_rate_limit():
                return

            _message = await ctx.load(f"It appears that [**video**]({attachment.url}) isn't viewable\n> Attempting to repair the video now..")
            async with ctx.typing():
                media = await attachment.read()
                with tempfile.TemporaryDirectory() as temp_dir:
                    filename = f"file{functions.hash(attachment.filename)}"
                    temp_file = os.path.join(temp_dir, f"{filename}.mov")
                    async with async_open(temp_file, "wb") as file:
                        await file.write(media)

                    try:
                        terminal = await asyncio.wait_for(
                            asyncio.create_subprocess_shell(f"cd {temp_dir} && ffmpeg -i {temp_file} {filename}.mp4 -nostats -loglevel 0"), timeout=25
                        )
                        stdout, stderr = await terminal.communicate()
                    except asyncio.TimeoutError:
                        return await ctx.warn(f"Couldn't repair the [**video**]({attachment.url}) - Timeout")

                    if not os.path.exists(f"{temp_dir}/{filename}.mp4"):
                        return await ctx.warn(f"Couldn't repair the [**video**]({attachment.url})!")

                    with contextlib.suppress(discord.HTTPException):
                        await _message.delete()

                    await message.reply(file=discord.File(f"{temp_dir}/{filename}.mp4"))

    # @commands.Cog.listener("on_user_message")
    # async def check_emojis(self, ctx: wock.Context, message: discord.Message):
    #     """Check if the message contains an emoji"""

    #     if not message.emojis:
    #         return

    #     bucket = self.buckets.get("emoji")
    #     async with bucket["lock"]:
    #         bucket["data"][message.guild.id].update(map(int, message.emojis))

    @commands.Cog.listener("on_user_message")
    async def check_highlights(self, ctx: wock.Context, message: discord.Message):
        """Check if the message contains a highlighted keyword"""

        if not message.content:
            return

        highlights = [
            highlight
            async for highlight in self.bot.db.fetchiter(
                "SELECT DISTINCT ON (user_id) * FROM highlight_words WHERE POSITION(word in $1) > 0",
                message.content.lower(),
            )
            if highlight["user_id"] != message.author.id
            and ctx.guild.get_member(highlight["user_id"])
            and ctx.channel.permissions_for(ctx.guild.get_member(highlight["user_id"])).view_channel
        ]
        if highlights:
            bucket = self.bot.buckets.get("highlights").get_bucket(message)
            if bucket.update_rate_limit():
                return

            for highlight in highlights:
                if not highlight.get("word") in message.content.lower() or (
                    highlight.get("strict") and not highlight.get("word") == message.content.lower()
                ):
                    continue
                if member := message.guild.get_member(highlight.get("user_id")):
                    self.bot.dispatch("highlight", message, highlight["word"], member)

    @commands.Cog.listener()
    async def on_highlight(self, message: discord.Message, keyword: str, member: discord.Member):
        """Send a notification to the member for the keyword"""

        if member in message.mentions:
            return

        if blocked_entities := await self.bot.db.fetch("SELECT entity_id FROM highlight_block WHERE user_id = $1", member.id):
            if message.author.id in blocked_entities:
                return
            elif message.channel.id in blocked_entities:
                return
            elif message.category.id in blocked_entities:
                return
            elif message.guild.id in blocked_entities:
                return

        try:
            await self.bot.wait_for("user_activity", check=lambda channel, user: message.channel == channel and user == member, timeout=10)
            return
        except asyncio.TimeoutError:
            pass

        embed = discord.Embed(
            color=functions.config_color("main"),
            url=message.jump_url,
            title=f"Highlight in {message.guild}",
            description=f"Keyword **{discord.utils.escape_markdown(keyword)}** said in {message.channel.mention}\n>>> ",
        )
        embed.set_author(
            name=message.author.display_name,
            icon_url=message.author.display_avatar,
        )

        messages = list()
        try:
            async for ms in message.channel.history(limit=3, before=message):
                if ms.id == message.id:
                    continue
                if not ms.content:
                    continue

                messages.append(
                    f"[{discord.utils.format_dt(ms.created_at, 'T')}] {discord.utils.escape_markdown(str(ms.author))}:"
                    f" {functions.shorten(discord.utils.escape_markdown(ms.content), 50)}"
                )

            messages.append(
                f"__[{discord.utils.format_dt(message.created_at, 'T')}]__ {discord.utils.escape_markdown(str(message.author))}:"
                f" {functions.shorten(discord.utils.escape_markdown(message.content).replace(keyword, f'__{keyword}__'), 50)}"
            )

            async for ms in message.channel.history(limit=2, after=message):
                if ms.id == message.id:
                    continue
                if not ms.content:
                    continue

                messages.append(
                    f"[{discord.utils.format_dt(ms.created_at, 'T')}] {discord.utils.escape_markdown(str(ms.author))}:"
                    f" {functions.shorten(discord.utils.escape_markdown(ms.content), 50)}"
                )

        except discord.Forbidden:
            pass
        embed.description += "\n".join(messages)

        try:
            await member.send(embed=embed)
        except discord.Forbidden:
            pass

    @commands.Cog.listener("on_message_delete")
    async def on_message_delete(self, message: discord.Message):
        """Save deleted messages to redis"""

        data = {
            "timestamp": message.created_at.timestamp(),
            "content": message.content,
            "embeds": [embed.to_dict() for embed in message.embeds[:8] if not embed.type == "image" and not embed.type == "video"],
            "attachments": [
                attachment.proxy_url
                for attachment in (message.attachments + list((embed.thumbnail or embed.image) for embed in message.embeds if embed.type == "image"))
            ],
            "stickers": [sticker.url for sticker in message.stickers],
            "author": {
                "id": message.author.id,
                "name": message.author.name,
                "discriminator": message.author.discriminator,
                "avatar": message.author.avatar.url if message.author.avatar else None,
            },
        }
        await self.bot.redis.ladd(
            f"deleted_messages:{functions.hash(message.channel.id)}",
            data,
            ex=60,
        )

    @commands.Cog.listener("on_message_edit")
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Save edited messages to redis"""

        data = {
            "timestamp": before.created_at.timestamp(),
            "content": before.content,
            "embeds": [embed.to_dict() for embed in before.embeds[:8] if not embed.type == "image" and not embed.type == "video"],
            "attachments": [
                attachment.proxy_url
                for attachment in (before.attachments + list((embed.thumbnail or embed.image) for embed in before.embeds if embed.type == "image"))
            ],
            "stickers": [sticker.url for sticker in before.stickers],
            "author": {
                "id": before.author.id,
                "name": before.author.name,
                "discriminator": before.author.discriminator,
                "avatar": before.author.avatar.url if before.author.avatar else None,
            },
        }
        await self.bot.redis.ladd(
            f"edited_messages:{functions.hash(before.channel.id)}",
            data,
            ex=60,
        )

    @commands.Cog.listener("on_raw_reaction_remove")
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        """Save removed reactions to redis"""

        data = {
            "timestamp": discord.utils.utcnow().timestamp(),
            "message": payload.message_id,
            "user": payload.user_id,
            "emoji": str(payload.emoji),
        }

        await self.bot.redis.ladd(
            f"removed_reactions:{functions.hash(payload.channel_id)}",
            data,
            ex=60,
        )

    @commands.Cog.listener("on_user_message")
    async def check_afk(self, ctx: wock.Context, message: discord.Message):
        """Check for AFK statuses"""

        if timestamp := await self.bot.db.fetchval(
            "SELECT timestamp FROM afk WHERE user_id = $1",
            message.author.id,
        ):
            await self.bot.db.execute("DELETE FROM afk WHERE user_id = $1", message.author.id)

            duration = discord.utils.utcnow() - timestamp

            return await ctx.neutral(
                f"Welcome back! You were away for **{humanize.time(duration)}**",
                emoji="👋🏾",
            )

        if len(message.mentions) == 1:
            user = message.mentions[0]
            if row := await self.bot.db.fetchrow(
                "SELECT message, timestamp FROM afk WHERE user_id = $1",
                user.id,
            ):
                duration = discord.utils.utcnow() - row.get("timestamp")
                return await ctx.neutral(
                    f"**{user}** is currently AFK: **{row.get('message')}** - {humanize.time(duration)} ago",
                    emoji="💤",
                )

    @commands.command(
        name="afk",
        usage="<message>",
        example="stroking my shi rn..",
        aliases=["away"],
    )
    async def afk(self, ctx: wock.Context, *, message: str = "AFK"):
        """Set an away status for when you're mentioned"""

        message = functions.shorten(message, 200)

        if await self.bot.db.execute(
            "INSERT INTO afk (user_id, message, timestamp) VALUES($1, $2, $3) ON CONFLICT(user_id) DO NOTHING",
            ctx.author.id,
            message,
            discord.utils.utcnow(),
        ):
            await ctx.approve(f"You're now **AFK** with the status: **{message}**")

    @commands.command(
        name="firstmessage",
        usage="<channel>",
        example="#chat",
        aliases=["firstmsg", "first"],
    )
    async def firstmessage(self, ctx: wock.Context, *, channel: discord.TextChannel = None):
        """View the first message in a channel"""

        channel = channel or ctx.channel

        async for message in channel.history(limit=1, oldest_first=True):
            break

        await ctx.neutral(
            f"Jump to the [**first message**]({message.jump_url}) by **{message.author}**",
            emoji="📝",
        )

    @commands.command(
        name="color",
        usage="(hex code)",
        example="#ff0000",
        aliases=["colour", "hex", "clr"],
    )
    async def color(self, ctx: wock.Context, *, color: wock.Color):
        """View information about a color"""

        embed = discord.Embed(
            color=color,
        )
        embed.set_author(name=color)

        embed.add_field(
            name="RGB",
            value=", ".join([str(x) for x in color.to_rgb()]),
            inline=True,
        )
        embed.add_field(
            name="INT",
            value=color.value,
            inline=True,
        )
        embed.set_image(url=("https://singlecolorimage.com/get/" + str(color).replace("#", "") + "/150x50"))
        await ctx.send(embed=embed)

    @commands.command(
        name="dominant",
        usage="(image)",
        example="dscord.com/chnls/999/..png",
        aliases=["extract"],
    )
    async def dominant(self, ctx: wock.Context, *, image: wock.ImageFinder = None):
        """Get the dominant color of an image"""

        image = image or await wock.ImageFinder.search(ctx)

        color = await functions.extract_color(
            self.bot.redis,
            image,
        )
        await ctx.neutral(f"The dominant color is **{color}**", emoji="🎨", color=color)

    @commands.command(
        name="google",
        usage="(query)",
        example="Hello in Spanish",
        aliases=["g", "search", "ggl"],
    )
    async def google(self, ctx: wock.Context, *, query: str):
        """Search for something on Google"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://notsobot.com/api/search/google",
                params=dict(
                    query=query.replace(" ", ""),
                    safe="true" if not ctx.channel.is_nsfw() else "false",
                ),
            )
            data = await response.json()

        if not data.get("total_result_count"):
            return await ctx.warn(f"Couldn't find any images for **{query}**")

        embed = discord.Embed(title=f"Google Search: {query}", description="")

        for card in data.get("cards", []):
            embed.description += f"**Rich Card Information:** `{card.get('title')}`\n"
            if card.get("description"):
                embed.description += f"{card.get('description')}\n"
            for field in card.get("fields"):
                embed.description += f"> **{field.get('name')}:** `{field.get('value')}`\n"
            for section in card.get("sections"):
                embed.description += f"> **{section.get('title')}:** `{section['fields'][0].get('name')}`\n"
            if card.get("image"):
                embed.set_image(url=card.get("image"))

        for entry in data.get("results")[:2] if data.get("cards", []) else data.get("results")[:3]:
            embed.add_field(
                name=entry.get("title"),
                value=f"{entry.get('url')}\n{entry.get('description')}",
                inline=False,
            )
        await ctx.send(embed=embed)

    @commands.command(
        name="image",
        usage="(query)",
        example="Clairo",
        aliases=["img", "im", "i"],
    )
    async def image(self, ctx: wock.Context, *, query: str):
        """Search Google for an image"""

        response = await self.bot.session.get(
            "https://notsobot.com/api/search/google/images",
            params=dict(
                query=query.replace(" ", ""),
                safe="true" if not ctx.channel.is_nsfw() else "false",
            ),
        )
        data = await response.json()

        if not data:
            return await ctx.warn(f"Couldn't find any images for **{query}**")

        entries = [
            discord.Embed(
                url=entry.get("url"),
                title=entry.get("header"),
                description=entry.get("description"),
            ).set_image(url=entry["image"]["url"])
            for entry in data
            if not entry.get("header") in ("TikTok", "Facebook")
        ]
        await ctx.paginate(entries)

    @commands.command(
        name="urban",
        usage="(query)",
        example="projecting",
        aliases=["urbandictionary", "ud"],
    )
    async def urban(self, ctx: wock.Context, *, query: str):
        """Search for a definition on Urban Dictionary"""

        response = await self.bot.session.get("http://api.urbandictionary.com/v0/define", params=dict(term=query))
        data = await response.json()

        if not data.get("list"):
            return await ctx.warn(f"Couldn't find any definitions for **{query}**")

        def repl(match):
            word = match.group(2)
            return f"[{word}](https://{word.replace(' ', '-')}.urbanup.com)"

        entries = [
            discord.Embed(
                url=entry.get("permalink"),
                title=entry.get("word"),
                description=re.compile(r"(\[(.+?)\])").sub(repl, entry.get("definition")),
            )
            .add_field(
                name="Example",
                value=re.compile(r"(\[(.+?)\])").sub(repl, entry.get("example")),
                inline=False,
            )
            .set_footer(text=f"👍 {entry.get('thumbs_up'):,} 👎 {entry.get('thumbs_down'):,} - {entry.get('author')}")
            for entry in data.get("list")
        ]
        await ctx.paginate(entries)

    @commands.command(name="github", usage="(username)", example="rxnk", aliases=["gh"])
    async def github(self, ctx: wock.Context, username: str):
        """Search for a user on GitHub"""

        response = await self.bot.session.get(
            "https://dev.wock.cloud/github/profile",
            params=dict(username=username),
            headers=dict(
                Authorization=self.bot.config["api"].get("wock"),
            ),
        )
        data = await response.json()

        if not data.get("user"):
            return await ctx.warn(f"Couldn't find a profile for **{username}**")
        user = data.get("user")
        repositories = data.get("repositories")

        embed = discord.Embed(
            url=user.get("url"),
            title=user.get("name") if not user.get("username") else f"{user.get('username')} (@{user.get('name')})",
            description=user.get("bio"),
        )

        if followers := user["statistics"].get("followers"):
            embed.add_field(
                name="Followers",
                value=f"{followers:,}",
                inline=True,
            )
        if following := user["statistics"].get("following"):
            embed.add_field(
                name="Following",
                value=f"{following:,}",
                inline=True,
            )
        if gists := user["statistics"].get("gists"):
            embed.add_field(
                name="Gists",
                value=f"{gists:,}",
                inline=True,
            )

        information = ""
        if user.get("location"):
            information += f"\n> 🌎 [**{user['location'].get('name')}**]({user['location'].get('url')})"
        if user.get("company"):
            information += f"\n> 🏢 **{user.get('company')}**"
        if user["connections"].get("website"):
            information += f"\n> 🌐 {user['connections'].get('website')}"
        if user["connections"].get("twitter"):
            information += f"\n> 🐦 **{user['connections'].get('twitter')}**"

        if information:
            embed.add_field(name="Information", value=information, inline=False)
        if repositories:
            embed.add_field(
                name=f"Repositories ({len(repositories)})",
                value="\n".join(
                    f"> [`⭐ {repo['statistics'].get('stars')},"
                    f" {datetime.fromisoformat(repo.get('created')).strftime('%m/%d/%y')} {repo.get('name')}`]({repo.get('url')})"
                    for repo in sorted(
                        repositories,
                        key=lambda repo: repo["statistics"].get("stars"),
                        reverse=True,
                    )[:3]
                ),
                inline=False,
            )
        embed.set_thumbnail(url=user.get("avatar"))
        embed.set_footer(
            text="Created",
            icon_url="https://cdn.discordapp.com/emojis/843537056541442068.png",
        )
        embed.timestamp = datetime.fromisoformat(user.get("created"))
        await ctx.send(embed=embed)

    @commands.command(
        name="lyrics",
        usage="(query)",
        example="Two Times by Destroy Lonely",
        aliases=["lyric", "lyr", "ly"],
    )
    async def lyrics(self, ctx: wock.Context, *, query: str):
        """Search for lyrics on Genius"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/genius/lyrics",
                params=dict(query=query),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Couldn't find any results for **{query}**")

        embed = discord.Embed(
            url=data.get("url"),
            title=f"{data.get('title')} by {data.get('artist')}",
            description=list(),
        )
        for lyric in data.get("lyrics").split("\n"):
            embed.description.append(lyric)

        embed.set_thumbnail(url=data.get("thumbnail"))
        embed.set_footer(
            text="Genius",
            icon_url="https://wock.cloud/assets/genius.png",
        )

        await ctx.paginate(embed, max_entries=15, counter=False)

    @commands.command(
        name="spotify",
        usage="(query)",
        example="25 Dollar Fanta",
        aliases=["sptrack", "sp"],
    )
    async def spotify(self, ctx: wock.Context, *, query: str):
        """Search for a song on Spotify"""

        results = await self.spotify_client.search_tracks(
            q=query,
            limit=1,
        )
        if not results:
            return await ctx.warn(f"Couldn't find any results for **{query}**")

        await ctx.reply(results[0].link)

    @commands.command(
        name="spotifyalbum",
        usage="(query)",
        example="The Life of Pablo",
        aliases=["spalbum", "spa"],
    )
    async def spotifyalbum(self, ctx: wock.Context, *, query: str):
        """Search for an album on Spotify"""

        results = await self.spotify_client.search_albums(
            q=query,
            limit=1,
        )
        if not results:
            return await ctx.warn(f"Couldn't find any results for **{query}**")

        await ctx.reply(results[0].link)

    @commands.command(
        name="soundcloud",
        usage="(query)",
        example="In Ha Mood by Ice Spice",
        aliases=["sc"],
    )
    async def soundcloud(self, ctx: wock.Context, *, query: str):
        """Search for a song on SoundCloud"""

        response = await self.bot.session.get(
            "https://dev.wock.cloud/soundcloud/search",
            params=dict(query=query),
            headers=dict(
                Authorization=self.bot.config["api"].get("wock"),
            ),
        )
        data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Couldn't find any results for **{query}**")

        await ctx.reply(data[0].get("url"))

    @commands.command(
        name="itunes",
        usage="(query)",
        example="Exchange by Bryson Tiller",
        aliases=["applemusic", "apple", "am"],
    )
    async def itunes(self, ctx: wock.Context, *, query: str):
        """Search for a song on iTunes"""

        response = await self.bot.session.get(
            "https://dev.wock.cloud/itunes/search",
            params=dict(query=query),
            headers=dict(
                Authorization=self.bot.config["api"].get("wock"),
            ),
        )
        data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Couldn't find any results for **{query}**")

        await ctx.reply(data[0].get("url"))

    @commands.group(
        name="snapchat",
        usage="(username)",
        example="daviddobrik",
        aliases=["snap"],
        invoke_without_command=True,
    )
    async def snapchat(self, ctx: wock.Context, username: str):
        """View a Snapchat profile"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/snapchat/profile",
                params=dict(username=username),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://www.snapchat.com/add/{functions.format_uri(username)}) not found")

        embed = discord.Embed(
            url=data.get("url"),
            title=f"{data.get('display_name')} on Snapchat",
            description=data.get("bio"),
        )

        if data.get("bitmoji"):
            embed.set_image(url=data.get("bitmoji"))
        else:
            embed.set_thumbnail(url=data.get("snapcode"))
        await ctx.send(embed=embed)

    @snapchat.command(
        name="stories",
        usage="(username)",
        example="daviddobrik",
        aliases=["story"],
    )
    async def snapchat_stories(self, ctx: wock.Context, username: str):
        """View public Snapchat stories"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/snapchat/profile",
                params=dict(username=username),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://www.snapchat.com/add/{functions.format_uri(username)}) not found")
        if not data.get("stories"):
            return await ctx.warn(f"Profile [**{username}**](https://www.snapchat.com/add/{functions.format_uri(username)}) has no public stories")

        entries = []
        for story in data.get("stories"):
            embed = discord.Embed(
                url=data.get("url"),
                title=f"{data.get('display_name')} on Snapchat",
            )

            if story["type"] == "image":
                embed.set_image(url=story.get("url"))
            else:
                embed.add_attachment((story.get("url"), f"wockSnapChat{tuuid.random()}.mp4"))

            entries.append(embed)

        await ctx.paginate(entries)

    @snapchat.command(
        name="highlights",
        usage="(username)",
        example="daviddobrik",
        aliases=["highlight"],
    )
    async def snapchat_highlights(self, ctx: wock.Context, username: str):
        """View public Snapchat highlights"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/snapchat/profile",
                params=dict(username=username),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://www.snapchat.com/add/{functions.format_uri(username)}) not found")
        if not data.get("highlights"):
            return await ctx.warn(f"Profile [**{username}**](https://www.snapchat.com/add/{functions.format_uri(username)}) has no public highlights")

        entries = [
            discord.Embed(
                url=data.get("url"),
                title=f"{data.get('display_name')} on Snapchat",
            ).set_image(url=highlight.get("url"))
            for highlight in data.get("highlights")
            if highlight["type"] == "image"
        ]
        await ctx.paginate(entries)

    @commands.command(name="pinterest", usage="(username)", example="@sentipedes", aliases=["pint", "pin"])
    async def pinterest(self, ctx: wock.Context, username: str):
        """View a Pinterest profile"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/pinterest/profile",
                params=dict(username=username),
                headers=dict(Authorization=self.bot.config["api"].get("wock")),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://pinterest.com/{functions.format_uri(username)}) not found")
        else:
            account = models.PinterestUser(**data)

        embed = discord.Embed(
            url=account.url,
            title=f"{account.display_name} (@{account.username})",
            description=account.bio,
        )

        for field, value in account.statistics.dict().items():
            embed.add_field(
                name=field.title(),
                value=f"{value:,}",
                inline=True,
            )
        embed.set_thumbnail(url=account.avatar)
        await ctx.send(embed=embed)

    @commands.command(name="weheartit", usage="(username)", example="@re93ka", aliases=["whi"])
    async def weheartit(self, ctx: wock.Context, username: str):
        """View a We Heart It profile"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/whi/profile",
                params=dict(username=username),
                headers=dict(Authorization=self.bot.config["api"].get("wock")),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://weheartit.com/{functions.format_uri(username)}) not found")
        else:
            account = models.WeHeartItUser(**data)

        embed = discord.Embed(
            url=account.url,
            title=f"{account.display_name} (@{account.username})",
            description=account.description,
        )

        for field, value in account.statistics.dict().items():
            if field == "posts":
                continue

            embed.add_field(
                name=field.title(),
                value=(f"{int(value):,}" if value.isdigit() else value),
                inline=True,
            )
        embed.set_thumbnail(url=account.avatar)
        await ctx.send(embed=embed)

    @commands.group(name="tiktok", usage="(username)", example="@kyliejenner", aliases=["tt"], invoke_without_command=True)
    async def tiktok(self, ctx: wock.Context, username: str):
        """View a TikTok profile"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/tiktok/profile",
                params=dict(username=username),
                headers=dict(Authorization=self.bot.config["api"].get("wock")),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://tiktok.com/@{functions.format_uri(username)}) not found")
        else:
            account = models.TikTokUser(**data)

        embed = discord.Embed(
            url=account.url,
            title=f"{account.nickname} (@{account.username}) {'☑️' if account.verified else ''}",
            description=account.signature,
        )

        for field, value in account.statistics.dict().items():
            embed.add_field(
                name=field.title(),
                value=(f"{int(value):,}" if value.isdigit() else value),
                inline=True,
            )
        embed.set_thumbnail(url=account.avatar)
        await ctx.send(embed=embed)

    @tiktok.command(
        name="download",
        usage="(username)",
        example="@kyliejenner",
        parameters={
            "amount": {
                "converter": int,
                "description": "The amount of videos to download",
                "default": 5,
                "minimum": 1,
                "maximum": 5,
                "aliases": ["a", "count", "c"],
            }
        },
        aliases=["dl", "videos", "vids"],
    )
    @commands.cooldown(1, 30, commands.BucketType.user)
    async def tiktok_download(self, ctx: wock.Context, username: str):
        """Download TikTok videos from a profile"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/tiktok/profile",
                params=dict(username=username),
                headers=dict(Authorization=self.bot.config["api"].get("wock")),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://tiktok.com/@{functions.format_uri(username)}) not found")
        else:
            account = models.TikTokUser(**data)

        if not account.videos:
            return await ctx.warn(f"Profile [**{account.nickname}**]({account.url}) has no public videos")

        amount = ctx.parameters.get("amount")
        message = await ctx.load(
            f"Downloading {functions.plural(account.videos[:amount], bold=True):video} from [**{account.nickname}**]({account.url})"
        )

        for video in account.videos[:amount]:
            self.bot.dispatch("message_repost", ctx, video.url, 0)
            await asyncio.sleep(1.5)

        await message.delete()

    # @tiktok.command(
    #     name="add",
    #     usage="(channel) (username)",
    #     example="#feed @kyliejenner",
    #     aliases=["create", "feed"],
    # )
    # @commands.has_permissions(manage_guild=True)
    # async def tiktok_add(self, ctx: wock.Context, channel: discord.TextChannel, username: str):
    #     """Stream posts from a TikTok profile"""

    #     async with ctx.typing():
    #         response = await self.bot.session.get(
    #             "https://dev.wock.cloud/tiktok/profile",
    #             params=dict(username=username),
    #             headers=dict(Authorization=self.bot.config["api"].get("wock")),
    #         )
    #         data = await response.json()

    #     if "error" in data:
    #         return await ctx.warn(f"Profile [**{username}**](https://tiktok.com/@{functions.format_uri(username)}) not found")
    #     else:
    #         account = models.TikTokUser(**data)

    #     if not account.videos:
    #         return await ctx.warn(f"Profile [**{account.nickname}**]({account.url}) has no public videos")

    #     channel_ids = await self.bot.db.fetchval("SELECT channel_ids FROM tiktok WHERE username = $1", account.username) or []
    #     if channel.id in channel_ids:
    #         return await ctx.warn(f"Profile [**{account.nickname}**]({account.url}) is already being streamed in {channel.mention}")

    #     channel_ids.append(channel.id)
    #     await self.bot.db.execute("INSERT INTO tiktok (username, post_id, channel_ids) VALUES($1, $2, $3) ON CONFLICT (username) DO UPDATE SET channel_ids = $3", account.username, account.videos[0].id, channel_ids)
    #     await ctx.approve(f"Now streaming posts from [**{account.nickname}**]({account.url}) in {channel.mention}")

    # @tiktok.command(
    #     name="remove",
    #     usage="(channel) (username)",
    #     example="#feed @kyliejenner",
    #     aliases=["delete", "del", "rm", "stop"],
    # )
    # @commands.has_permissions(manage_guild=True)
    # async def tiktok_remove(self, ctx: wock.Context, channel: discord.TextChannel, username: str):
    #     """Remove a TikTok feed for a channel"""

    #     async with ctx.typing():
    #         response = await self.bot.session.get(
    #             "https://dev.wock.cloud/tiktok/profile",
    #             params=dict(username=username),
    #             headers=dict(Authorization=self.bot.config["api"].get("wock")),
    #         )
    #         data = await response.json()

    #     if "error" in data:
    #         return await ctx.warn(f"Profile [**{username}**](https://tiktok.com/@{functions.format_uri(username)}) not found")
    #     else:
    #         account = models.TikTokUser(**data)

    #     channel_ids = await self.bot.db.fetchval("SELECT channel_ids FROM tiktok WHERE username = $1", account.username) or []
    #     if not channel.id in channel_ids:
    #         return await ctx.warn(f"Profile [**{account.nickname}**]({account.url}) isn't being streamed in {channel.mention}")

    #     channel_ids.remove(channel.id)
    #     await self.bot.db.execute("UPDATE tiktok SET channel_ids = $2 WHERE username = $1", account.username, channel_ids)
    #     await ctx.approve(f"Stopped streaming posts from [**{account.nickname}**]({account.url}) in {channel.mention}")

    # @tiktok.command(
    #     name="list",
    #     aliases=["show", "all"],
    # )
    # @commands.has_permissions(manage_guild=True)
    # async def tiktok_list(self, ctx: wock.Context):
    #     """View all TikTok feeds"""

    #     # feeds = [
    #     #     f"{channel.mention} - [**@{row.get('username')}**](https://tiktok.com/@{row.get('username')})"
    #     #     async for row in self.bot.db.fetchiter(
    #     #         "SELECT username, array_agg(channel_id) AS channel_ids FROM tiktok WHERE ANY($1 IN channel_ids) GROUP BY username",
    #     #     )
    #     # ]
    #     await ctx.reply("I haven't finished this :/")

    @commands.command(name="cashapp", usage="(username)", example="madeitsick", aliases=["ca"])
    async def cashapp(self, ctx: wock.Context, username: str):
        """View a Cash App profile"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/cashapp/profile",
                params=dict(username=username),
                headers=dict(Authorization=self.bot.config["api"].get("wock")),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile [**{username}**](https://cash.app/{functions.format_uri(username)}) not found")
        else:
            account = models.CashApp(**data)

        embed = discord.Embed(
            color=discord.Color.from_str(account.avatar.accent_color),
            url=account.url,
            title=f"{account.display_name} ({account.cashtag})",
        )

        embed.set_thumbnail(url=account.avatar.image_url)
        embed.set_image(url=account.qr)
        await ctx.send(embed=embed)

    # @commands.command(
    #     name="roblox", usage="(username)", example="rxflipflop", aliases=["rblx", "rbx"]
    # )
    # async def roblox(self, ctx: wock.Context, username: str):
    #     """View information about a Roblox user"""

    #     await ctx.typing()
    #     response = await self.bot.session.get(
    #         "https://api.roblox.com/users/get-by-username",
    #         params=dict(username=username),
    #     )
    #     data = await response.json()

    #     if data.get("errorMessage"):
    #         return await ctx.warn(f"Profile **{username}** not found")

    #     response = await self.bot.session.get(
    #         "https://users.roblox.com/v1/users/" + str(data.get("Id"))
    #     )
    #     data = await response.json()

    #     embed = discord.Embed(
    #         url=f"https://www.roblox.com/users/{data.get('id')}/profile",
    #         title=f"{data.get('displayName')} (@{data.get('name')})",
    #         description=data.get("description"),
    #     )

    #     embed.add_field(
    #         name="Created",
    #         value=discord.utils.format_dt(
    #             datetime.fromisoformat(
    #                 data.get("created")
    #                 .replace("Z", "+00:00")
    #                 .replace("T", " ")
    #                 .split(".")[0]
    #                 .replace(" ", "T")
    #             ),
    #             style="R",
    #         ),
    #         inline=True,
    #     )
    #     embed.set_thumbnail(
    #         url=f"https://www.roblox.com/headshot-thumbnail/image?userId={data.get('id')}&width=700&height=700&format=png"
    #     )
    #     await ctx.send(embed=embed)

    @commands.command(
        name="minecraft",
        usage="(username)",
        example="DestroyMeowly",
        aliases=["namemc"],
    )
    async def minecraft(self, ctx: wock.Context, username: str):
        """View a Minecraft profile"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/minecraft/profile",
                params=dict(username=username),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn(f"Profile **{username}** not found")

        embed = discord.Embed(
            url=f"https://namemc.com/profile/{data.get('uuid')}",
            title=data.get("username"),
        )

        if name_history := data.get("name_history"):
            embed.add_field(
                name="Name History",
                value="\n".join(
                    f"{name} ({discord.utils.format_dt(datetime.fromtimestamp(timestamp), style='R')})" for name, timestamp in name_history
                ),
                inline=False,
            )
        embed.set_image(url=data["meta"].get("body"))
        await ctx.send(embed=embed)

    @commands.command(
        name="valorant",
        usage="(username)",
        example="rxsoulja#1337",
        aliases=["valo", "val"],
    )
    @commands.cooldown(3, 30, commands.BucketType.member)
    async def valorant(self, ctx: wock.Context, *, username: str):
        """View information about a Valorant Player"""

        sliced = username.split("#", 1)
        if not len(sliced) == 2:
            return await ctx.send_help()
        else:
            username, tag = sliced

        await ctx.load(f"Searching for `{username}#{tag}`")

        response = await self.bot.session.get(
            f"https://api.henrikdev.xyz/valorant/v1/account/{functions.format_uri(username)}/{functions.format_uri(tag)}",
            headers=dict(
                Authorization=self.bot.config["api"].get("henrik"),
            ),
        )
        if response.status == 404:
            return await ctx.warn(f"Couldn't find an account for `{username}#{tag}`")
        elif response.status == 429:
            return await ctx.warn(f"The **API** is currently **rate limited** - Try again later")
        else:
            data = await response.json()
            if not "data" in data:
                return await ctx.warn(f"Couldn't find an account for `{username}#{tag}`")

            response = await self.bot.session.get(
                f"https://api.henrikdev.xyz/valorant/v2/mmr/{data['data']['region']}/{functions.format_uri(username)}/{functions.format_uri(tag)}",
                headers=dict(
                    Authorization=self.bot.config["api"].get("henrik"),
                ),
            )
            if response.status == 404:
                return await ctx.warn(f"Couldn't find an account for `{username}#{tag}`")
            elif response.status == 429:
                return await ctx.warn(f"The **API** is currently **rate limited** - Try again later")
            else:
                _data = await response.json()

            account = models.ValorantAccount(
                region=data["data"]["region"].upper(),
                username=(data["data"]["name"] + "#" + data["data"]["tag"]),
                level=data["data"]["account_level"],
                rank=_data["data"]["current_data"]["currenttierpatched"] or "Unranked",
                elo=_data["data"]["current_data"]["elo"] or 0,
                elo_change=_data["data"]["current_data"]["mmr_change_to_last_game"] or 0,
                card=data["data"]["card"]["small"],
                updated_at=data["data"]["last_update_raw"],
            )

        response = await self.bot.session.get(
            f"https://api.henrikdev.xyz/valorant/v3/matches/{account.region}/{functions.format_uri(username)}/{functions.format_uri(tag)}",
            params=dict(filter="competitive"),
            headers=dict(
                Authorization=self.bot.config["api"].get("henrik"),
            ),
        )
        if response.status == 404:
            return await ctx.warn(f"Couldn't find any matches for `{username}#{tag}`")
        elif response.status == 429:
            return await ctx.warn(f"The **API** is currently **rate limited** - Try again later")
        else:
            data = await response.json()
            matches = [
                models.ValorantMatch(
                    map=match["metadata"]["map"],
                    rounds=match["metadata"]["rounds_played"],
                    status=("Victory" if match["teams"]["red"]["has_won"] else "Defeat"),
                    kills=match["players"]["all_players"][0]["stats"]["kills"],
                    deaths=match["players"]["all_players"][0]["stats"]["deaths"],
                    started_at=match["metadata"]["game_start"],
                )
                for match in data["data"]
            ]

        embed = discord.Embed(
            url=f"https://tracker.gg/valorant/profile/riot/{functions.format_uri(account.username)}/overview",
            title=f"{account.region}: {account.username}",
            description=(
                f">>> **Account Level:** {account.level}\n**Rank & ELO:** {account.rank} &"
                f" {account.elo} (`{'+' if account.elo_change >= 1 else ''}{account.elo_change}`)"
            ),
        )

        if matches:
            embed.add_field(
                name="Competitive Matches",
                value="\n".join(
                    f"> {discord.utils.format_dt(match.started_at, 'd')} {match.status} (`{f'+{match.kills}' if match.kills >= match.deaths else f'-{match.deaths}'}`)"
                    for match in matches
                ),
            )
        embed.set_thumbnail(
            url=account.card,
        )
        embed.set_footer(
            text="Last Updated",
            icon_url="https://img.icons8.com/color/512/valorant.png",
        )
        embed.timestamp = account.updated_at
        await ctx.send(embed=embed)
        with contextlib.suppress(discord.HTTPException):
            await ctx.previous_load.delete()

    @commands.group(
        name="fortnite",
        usage="(subcommand) <args>",
        example="lookup Nog Ops",
        aliases=["fort", "fn"],
        invoke_without_command=True,
    )
    async def fortnite(self, ctx: wock.Context):
        """Fortnite cosmetic commands"""

        await ctx.send_help()

    @fortnite.command(name="shop", aliases=["store"])
    async def fortnite_shop(self, ctx: wock.Context):
        """View the current Fortnite item shop"""

        embed = discord.Embed(
            title="Fortnite Item Shop",
        )

        embed.set_image(url=f"https://bot.fnbr.co/shop-image/fnbr-shop-{discord.utils.utcnow().strftime('%-d-%-m-%Y')}.png")
        await ctx.send(embed=embed)

    @fortnite.command(name="lookup", usage="(cosmetic)", example="Nog Ops", aliases=["search", "find"])
    async def fortnite_lookup(self, ctx: wock.Context, *, cosmetic: str):
        """Search for a cosmetic with the last release dates"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://fortnite-api.com/v2/cosmetics/br/search",
                params=dict(
                    name=cosmetic,
                    matchMethod="contains",
                ),
                headers=dict(Authorization=self.bot.config["api"].get("fortnite")),
            )
            data = await response.json()

        if not data.get("data"):
            return await ctx.warn(f"Couldn't find any cosmetics matching **{cosmetic}**\n> Search for a cosmetic [**here**](https://fnbr.co/list)")
        else:
            cosmetic = data.get("data")

        embed = discord.Embed(
            url=f"https://fnbr.co/{cosmetic['type'].get('value')}/{cosmetic.get('name').replace(' ', '-')}",
            title=cosmetic.get("name"),
            description=f"{cosmetic.get('description')}\n> {cosmetic['introduction'].get('text').replace('Chapter 1, ', '')}",
        )

        if cosmetic.get("shopHistory"):
            embed.add_field(
                name="Release Dates",
                value="\n".join(
                    f"{discord.utils.format_dt(datetime.fromisoformat(date.replace('Z', '+00:00').replace('T', ' ').split('.')[0].replace(' ', 'T')), style='D')} ({discord.utils.format_dt(datetime.fromisoformat(date.replace('Z', '+00:00').replace('T', ' ').split('.')[0].replace(' ', 'T')), style='R')})"
                    for date in list(reversed(cosmetic.get("shopHistory")))[:5]
                ),
                inline=False,
            )
        else:
            embed.add_field(
                name="Release Date",
                value=(
                    f"{discord.utils.format_dt(datetime.fromisoformat(cosmetic.get('added').replace('Z', '+00:00').replace('T', ' ').split('.')[0].replace(' ', 'T')), style='D')} ({discord.utils.format_dt(datetime.fromisoformat(cosmetic.get('added').replace('Z', '+00:00').replace('T', ' ').split('.')[0].replace(' ', 'T')), style='R')})"
                ),
                inline=False,
            )
        embed.set_thumbnail(url=cosmetic["images"].get("icon"))
        await ctx.send(embed=embed)

    # @commands.command(
    #     name="correction",
    #     usage="(text)",
    #     example="wats up fam",
    #     aliases=["correct", "grammar"],
    # )
    # async def correction(self, ctx: wock.Context, *, text: str):
    #     """Corrects grammar mistakes"""

    #     await ctx.typing()
    #     response = await self.bot.session.get(
    #         "https://dev.wock.cloud/grammar/correction",
    #         params=dict(text=text),
    #     )
    #     data = await response.json()

    #     if data.get("modified"):
    #         await ctx.send(
    #             data.get("corrected"), allowed_mentions=discord.AllowedMentions.none()
    #         )
    #     else:
    #         await ctx.warn("There aren't any **grammar mistakes**")

    @commands.command(
        name="dictionary",
        usage="(word)",
        example="fam",
        aliases=["definition", "define"],
    )
    async def dictionary(self, ctx: wock.Context, *, word: str):
        """View the definition of a word"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/dictionary/define",
                params=dict(word=word),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if not data.get("word"):
            return await ctx.warn(f"Couldn't find a definition for **{word}**")

        embed = discord.Embed(
            url=data.get("url"),
            title=data.get("word"),
            description=(
                f"[`{data['meaning'].get('pronunciation') or data['meaning'].get('speech')}`]({data['meaning'].get('pronunciation_url', '')})\n>"
                f" {data['meaning'].get('definition')}"
            ),
        )
        if data["meaning"].get("example"):
            embed.add_field(
                name="Example",
                value=data["meaning"].get("example"),
                inline=False,
            )

        await ctx.send(embed=embed)

    @commands.group(
        name="rtfm",
        usage="(entity)",
        example="commands.Context",
        aliases=["rtfd"],
        invoke_without_command=True,
    )
    async def _rtfm(self, ctx: wock.Context, *, entity: str):
        """View documentation for a discord.py entity"""

        results = await rtfm.search(ctx, "discord.py", entity)
        if results:
            embed = discord.Embed(description="\n".join(f"[`{entity}`]({url})" for entity, url in results))

            await ctx.send(embed=embed)
        else:
            await ctx.warn(f"Couldn't find an entity for **{entity}**")

    @_rtfm.command(name="python", usage="(entity)", example="asyncio.gather", aliases=["py"])
    async def _rtfm_python(self, ctx: wock.Context, *, entity: str):
        """View documentation for a python entity"""

        results = await rtfm.search(ctx, "python", entity)
        if results:
            embed = discord.Embed(description="\n".join(f"[`{entity}`]({url})" for entity, url in results))

            await ctx.send(embed=embed)
        else:
            await ctx.warn(f"Couldn't find an entity for **{entity}**")

    @_rtfm.command(name="refresh", hidden=True)
    @commands.is_owner()
    async def _rtfm_refresh(self, ctx: wock.Context):
        """Refreshes the rtfm cache"""

        async with ctx.typing():
            await rtfm.build_rtfm_lookup_table(ctx)
        await ctx.approve("Refreshed the **rtfm cache**")

    @commands.command(
        name="nba",
    )
    async def nba(self, ctx: wock.Context):
        """National Basketball Association Scores"""

        scores = await self.sport_scores("basketball/nba")
        await ctx.paginate(scores)

    @commands.command(
        name="nfl",
    )
    async def nfl(self, ctx: wock.Context):
        """National Football League Scores"""

        scores = await self.sport_scores("football/nfl")
        await ctx.paginate(scores)

    @commands.command(
        name="mlb",
    )
    async def mlb(self, ctx: wock.Context):
        """Major League Baseball Scores"""

        scores = await self.sport_scores("baseball/mlb")
        await ctx.paginate(scores)

    @commands.command(
        name="nhl",
    )
    async def nhl(self, ctx: wock.Context):
        """National Hockey League Scores"""

        scores = await self.sport_scores("hockey/nhl")
        await ctx.paginate(scores)

    @commands.group(
        name="btc",
        usage="(address)",
        example="bc1qe5vaz29nw0zkyayep..",
        aliases=["bitcoin"],
        invoke_without_command=True,
    )
    async def btc(self, ctx: wock.Context, address: str):
        """View information about a bitcoin address"""

        response = await self.bot.session.get(
            "https://blockchain.info/rawaddr/" + str(address),
        )
        data = await response.json()

        if data.get("error"):
            return await ctx.warn(f"Couldn't find an **address** for `{address}`")

        response = await self.bot.session.get(
            "https://min-api.cryptocompare.com/data/price",
            params=dict(fsym="BTC", tsyms="USD"),
        )
        price = await response.json()
        price = price["USD"]

        embed = discord.Embed(
            url=f"https://mempool.space/address/{address}",
            title="Bitcoin Address",
        )

        embed.add_field(
            name="Balance",
            value=f"{(data['final_balance'] / 100000000 * price):,.2f} USD",
        )
        embed.add_field(
            name="Received",
            value=f"{(data['total_received'] / 100000000 * price):,.2f} USD",
        )
        embed.add_field(name="Sent", value=f"{(data['total_sent'] / 100000000 * price):,.2f} USD")
        if data["txs"]:
            embed.add_field(
                name="Transactions",
                value="\n".join(
                    f"> [`{tx['hash'][:19]}..`](https://mempool.space/tx/{tx['hash']}) {(tx['result'] / 100000000 * price):,.2f} USD"
                    for tx in data["txs"][:5]
                ),
            )

        await ctx.send(embed=embed)

    @btc.command(
        name="subscribe",
        usage="(transaction)",
        example="2083b2e0e3983882755cc..",
        aliases=["sub", "notify", "watch"],
    )
    @checks.require_dm()
    async def btc_subscribe(self, ctx: wock.Context, transaction: str):
        """Send a notification when a transaction is confirmed"""

        response = await self.bot.session.get(
            "https://mempool.space/api/tx/" + str(transaction) + "/status",
        )
        if response.status != 200:
            return await ctx.warn(f"Couldn't find a **transaction** for [`{functions.shorten(transaction)}`](https://mempool.space/tx/{transaction})")

        data = await response.json()
        if data.get("confirmed"):
            return await ctx.warn(
                f"Transaction [`{functions.shorten(transaction)}`](https://mempool.space/tx/{transaction}) already has a **confirmation**"
            )

        try:
            await self.bot.db.execute(
                "INSERT INTO btc_subscriptions (user_id, transaction) VALUES ($1, $2)",
                ctx.author.id,
                transaction.upper(),
            )
        except:
            await ctx.warn(f"Already subscribed to [`{functions.shorten(transaction)}`](https://mempool.space/tx/{transaction})")
        else:
            await ctx.approve(f"Subscribed to [`{functions.shorten(transaction)}`](https://mempool.space/tx/{transaction})")

    # @commands.command(
    #     name="gas",
    # )
    # async def gas(self, ctx: wock.Context):
    #     """View the current Ethereum gas prices"""

    #     await ctx.typing()
    #     response = await self.bot.session.get(
    #         "https://api.owlracle.info/v3/eth/gas",
    #     )
    #     data = await response.json()

    #     embed = discord.Embed(
    #         title="Ethereum Gas Prices",
    #     )

    #     embed.add_field(
    #         name="Slow",
    #         value=f"**GWEI:** {data['speeds'][0]['maxFeePerGas']:,.2f}\n**FEE:** ${data['speeds'][0]['estimatedFee']:,.2f} USD",
    #         inline=True,
    #     )
    #     embed.add_field(
    #         name="Standard",
    #         value=f"**GWEI:** {data['speeds'][1]['maxFeePerGas']:,.2f}\n**FEE:** ${data['speeds'][1]['estimatedFee']:,.2f} USD",
    #         inline=True,
    #     )
    #     embed.add_field(
    #         name="Fast",
    #         value=f"**GWEI:** {data['speeds'][2]['maxFeePerGas']:,.2f}\n**FEE:** ${data['speeds'][2]['estimatedFee']:,.2f} USD",
    #         inline=True,
    #     )
    #     embed.set_footer(
    #         text="OwlRacle",
    #         icon_url="https://wock.cloud/assets/static/owlracle.png",
    #     )
    #     embed.timestamp = functions.get_timestamp(data["timestamp"])
    #     await ctx.send(embed=embed)

    @commands.command(
        name="translate",
        usage="<language> (text)",
        example="Spanish Hello!",
        aliases=["tr"],
    )
    async def translate(self, ctx: wock.Context, language: Optional[wock.Language] = "en", *, text: str):
        """Translate text to another language"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://clients5.google.com/translate_a/single",
                params={"dj": "1", "dt": ["sp", "t", "ld", "bd"], "client": "dict-chrome-ex", "sl": "auto", "tl": language, "q": text},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
                },
            )
            if response.status != 200:
                return await ctx.warn("Couldn't **translate** the **text**")

            data = await response.json()
            text = "".join(sentence.get("trans", "") for sentence in data.get("sentences", []))
            if not text:
                return await ctx.warn("Couldn't **translate** the **text**")

        if ctx.author.mobile_status != discord.Status.offline:
            return await ctx.reply(text)
        else:
            embed = discord.Embed(
                title="Google Translate",
                description=f"```{text[:4000]}```",
            )
            await ctx.reply(embed=embed)

    @commands.group(
        name="ocr",
        usage="(image)",
        example="dscord.com/chnls/999/..png",
        aliases=["read", "text"],
        invoke_without_command=True,
    )
    async def ocr(self, ctx: wock.Context, *, image: wock.ImageFinderStrict = None):
        """Extract text from an image"""

        image = image or await wock.ImageFinderStrict.search(ctx)

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/google/ocr",
                params=dict(content=image),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

        if "error" in data:
            return await ctx.warn("Couldn't detect any **text** in the **image**")

        if ctx.author.mobile_status != discord.Status.offline:
            return await ctx.reply(data["text"])
        else:
            embed = discord.Embed(
                title="Optical Character Recognition",
                description=f"```{data['text'][:4000]}```",
            )
            await ctx.reply(embed=embed)

    @ocr.command(name="translate", usage="<language> (image)", example="en dscord.com/chnls/999/..png", aliases=["tr"])
    async def ocr_translate(self, ctx: wock.Context, language: Optional[wock.Language] = "en", *, image: wock.ImageFinderStrict = None):
        """Translate text from an image"""

        image = image or await wock.ImageFinderStrict.search(ctx)

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://dev.wock.cloud/google/ocr",
                params=dict(content=image),
                headers=dict(
                    Authorization=self.bot.config["api"].get("wock"),
                ),
            )
            data = await response.json()

            if "error" in data:
                return await ctx.warn("Couldn't detect any **text** in the **image**")

            response = await self.bot.session.get(
                "https://clients5.google.com/translate_a/single",
                params={"dj": "1", "dt": ["sp", "t", "ld", "bd"], "client": "dict-chrome-ex", "sl": "auto", "tl": language, "q": data.get("text")},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
                },
            )
            if response.status != 200:
                return await ctx.warn("Couldn't **translate** the **text**")

            data = await response.json()
            text = "".join(sentence.get("trans", "") for sentence in data.get("sentences", []))
            if not text:
                return await ctx.warn("Couldn't **translate** the **text**")

        if ctx.author.mobile_status != discord.Status.offline:
            return await ctx.reply(text)
        else:
            embed = discord.Embed(
                title="Optical Character Recognition",
                description=f"```{text[:4000]}```",
            )
            await ctx.reply(embed=embed)

    @commands.command(
        name="transcript",
        usage="(video or audio)",
        example="dscord.com/chnls/999/..mp4",
        aliases=["transcribe"],
    )
    async def transcript(self, ctx: wock.Context, *, media: wock.MediaFinder = None):
        """Transcribe speech to text"""

        media = media or await wock.MediaFinder.search(ctx)

        async with ctx.typing():
            response = await self.bot.session.get(media)
            if sys.getsizeof(response.content) > 26214400:
                return await ctx.warn("Media is too large to **transcribe** (max 25MB)")

            media = await response.read()
            data = io.BytesIO(media)
            data.name = "file." + regex.MEDIA_URL.match(str(response.url)).group("mime")
            try:
                response = await openai.Audio.atranscribe(
                    model="whisper-1",
                    file=data,
                )
            except:
                return await ctx.warn(f"Couldn't **transcribe** media - Invalid format")

        if not response.get("text"):
            return await ctx.warn("Couldn't **transcribe** audio")

        await ctx.reply(response.get("text"), allowed_mentions=discord.AllowedMentions.none())

    @commands.command(
        name="shazam",
        usage="(video or audio)",
        example="dscord.com/chnls/999/..mp4",
        aliases=["identify"],
    )
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def shazam(self, ctx: wock.Context, *, media: wock.MediaFinder = None):
        """Identify a song from audio"""

        media = media or await wock.MediaFinder.search(ctx)

        async with ctx.typing():
            response = await self.bot.session.get(media)
            if sys.getsizeof(response.content) > 26214400:
                return await ctx.warn("Media is too large to **identify** (max 25MB)")

            media = await response.read()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(
                    temp_dir, f"file{functions.hash(str(response.url))}." + regex.MEDIA_URL.match(str(response.url)).group("mime")
                )
                async with async_open(temp_file, "wb") as file:
                    await file.write(media)

                try:
                    songrec = await asyncio.wait_for(
                        asyncio.create_subprocess_shell(
                            f'songrec audio-file-to-recognized-song "{temp_file}"',
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        ),
                        timeout=7,
                    )
                    stdout, stderr = await songrec.communicate()
                except asyncio.TimeoutError:
                    return await ctx.warn("Couldn't **recognize** the song - Timeout")

            try:
                output = json.loads(stdout.decode())
            except json.JSONDecodeError:
                return await ctx.warn("Couldn't **recognize** the song")

            if track := output.get("track", {}):
                await ctx.neutral(f"Found [**{track.get('title')}**]({track.get('url')}) by **{track.get('subtitle')}**", emoji="🎵")
            else:
                await ctx.warn("Couldn't **recognize** the song")

    @commands.command(
        name="synth",
        usage="<engine> (text)",
        example="ghostface hey mommy",
        aliases=["synthesizer", "synthesize", "tts"],
    )
    async def synth(self, ctx: wock.Context, engine: Optional[wock.SynthEngine], *, text: str):
        """Synthesize text into speech"""

        async with ctx.typing():
            response = await self.bot.session.post(
                "https://api16-normal-useast5.us.tiktokv.com/media/api/text/speech/invoke/",
                params=dict(
                    text_speaker=engine or "en_us_002",
                    req_text=text.replace("+", "plus").replace("-", "minus").replace("=", "equals").replace("/", "slash").replace("@", "at")[:300],
                    speaker_map_type=0,
                    aid=1233,
                ),
                headers={
                    "User-Agent": "com.zhiliaoapp.musically/2022600030 (Linux; U; Android 7.1.2; es_ES; SM-G988N; Build/NRD90M;tt-ok/3.12.13.1)",
                    "Cookie": "sessionid=" + self.bot.config["api"].get("tiktok"),
                },
            )
            data = await response.json()

        if data["status_code"] != 0:
            return await ctx.warn("Couldn't **synthesize** text")

        vstr: str = data["data"]["v_str"]
        _padding = len(vstr) % 4
        vstr = vstr + ("=" * _padding)

        decoded = base64.b64decode(vstr)
        clean_data = io.BytesIO(decoded)
        clean_data.seek(0)

        file = discord.File(fp=clean_data, filename=f"Synthesize{tuuid.random()}.mp3")
        await ctx.reply(file=file)

    @commands.command(
        name="wolfram",
        usage="(query)",
        example="integral of x^2",
        aliases=["wolframalpha", "wa", "w"],
    )
    async def wolfram(self, ctx: wock.Context, *, query: str):
        """Search a query on Wolfram Alpha"""

        async with ctx.typing():
            response = await self.bot.session.get(
                "https://notsobot.com/api/search/wolfram-alpha",
                params=dict(query=query),
            )
            data = await response.json()

        if not data.get("fields"):
            return await ctx.warn("Couldn't **understand** your input")

        embed = discord.Embed(
            url=data.get("url"),
            title=query,
        )

        for index, field in enumerate(data.get("fields")[:4]):
            if index == 2:
                continue

            embed.add_field(
                name=field.get("name"),
                value=(">>> " if index == 3 else "") + field.get("value").replace("( ", "(").replace(" )", ")").replace("(", "(`").replace(")", "`)"),
                inline=(False if index == 3 else True),
            )
        embed.set_footer(
            text="Wolfram Alpha",
            icon_url="https://wock.cloud/assets/wolfram-alpha.png",
        )
        await ctx.send(embed=embed)

    @commands.group(
        name="remind",
        usage="(duration) (text)",
        example=f"4h Kill My Family",
        aliases=["reminder"],
        invoke_without_command=True,
    )
    @checks.require_dm()
    async def remind(self, ctx: wock.Context, duration: wock.TimeConverter, *, text: str):
        """Set a reminder to do something"""

        if duration.seconds < 60:
            return await ctx.warn("Duration must be at least **1 minute**")

        try:
            await self.bot.db.execute(
                "INSERT INTO reminders (user_id, text, jump_url, created_at, timestamp) VALUES ($1, $2, $3, $4, $5)",
                ctx.author.id,
                text,
                ctx.message.jump_url,
                ctx.message.created_at,
                ctx.message.created_at + duration.delta,
            )
        except:
            await ctx.warn(f"Already being reminded for **{text}**")
        else:
            await ctx.approve(f"I'll remind you {discord.utils.format_dt(ctx.message.created_at + duration.delta, style='R')}")

    @remind.command(
        name="remove",
        usage="(text)",
        example="Kill My Family",
        aliases=["delete", "del", "rm", "cancel"],
    )
    async def remove(self, ctx: wock.Context, *, text: str):
        """Remove a reminder"""

        try:
            await self.bot.db.execute(
                "DELETE FROM reminders WHERE user_id = $1 AND lower(text) = $2",
                ctx.author.id,
                text.lower(),
                raise_exceptions=True,
            )
        except:
            await ctx.warn(f"Coudn't find a reminder for **{text}**")
        else:
            await ctx.approve(f"Removed reminder for **{text}**")

    @remind.command(
        name="list",
        aliases=["show", "view"],
    )
    async def reminders(self, ctx: wock.Context):
        """View your pending reminders"""

        reminders = await self.bot.db.fetch(
            "SELECT * FROM reminders WHERE user_id = $1",
            ctx.author.id,
        )
        if not reminders:
            return await ctx.warn("You don't have any **reminders**")

        await ctx.paginate(
            discord.Embed(
                title="Reminders",
                description=list(
                    f"**{functions.shorten(reminder['text'])}** ({discord.utils.format_dt(reminder['timestamp'], style='R')})"
                    for reminder in reminders
                ),
            )
        )

    @commands.command(
        name="seen",
        usage="(member)",
        example="rx#1337",
        aliases=["lastseen"],
    )
    async def seen(self, ctx: wock.Context, *, member: wock.Member):
        """View when a member was last seen"""

        last_seen = await self.bot.db.fetchval(
            "SELECT timestamp FROM metrics.seen WHERE user_id = $1",
            member.id,
        )
        if not last_seen:
            return await ctx.warn(f"**{member}** hasn't been **seen** yet")

        last_seen = datetime.fromtimestamp(last_seen)
        await ctx.neutral(f"**{member}** was **last seen** {discord.utils.format_dt(last_seen, style='R')}")

    @commands.command(
        name="namehistory",
        usage="<user>",
        example="rx#1337",
        aliases=["names", "nh"],
    )
    async def namehistory(self, ctx: wock.Context, *, user: wock.Member | wock.User = None):
        """View a user's name history"""

        user = user or ctx.author

        names = await self.bot.db.fetch(
            "SELECT name, timestamp FROM metrics.names WHERE user_id = $1 ORDER BY timestamp DESC",
            user.id,
        )
        if not names:
            return await ctx.warn(
                "You don't have any **names** in the database" if user == ctx.author else f"**{user}** doesn't have any **names** in the database"
            )

        await ctx.paginate(
            discord.Embed(
                title="Name History",
                description=list(f"**{name['name']}** ({discord.utils.format_dt(name['timestamp'], style='R')})" for name in names),
            )
        )

    @commands.command(
        name="avatarhistory",
        usage="<user>",
        example="rx#1337",
        aliases=["avatars", "avh", "avs", "ah"],
    )
    @commands.max_concurrency(1, commands.BucketType.user)
    @commands.cooldown(3, 30, commands.BucketType.user)
    async def avatarhistory(self, ctx: wock.Context, *, user: wock.Member | wock.User = None):
        """View a user's avatar history"""

        user = user or ctx.author

        avatars = await self.bot.db.fetch(
            "SELECT avatar, timestamp FROM metrics.avatars WHERE user_id = $1 ORDER BY timestamp DESC",
            user.id,
        )
        if not avatars:
            return await ctx.warn(
                "You don't have any **avatars** in the database" if user == ctx.author else f"**{user}** doesn't have any **avatars** in the database"
            )

        async with ctx.typing():
            image = await functions.collage([row.get("avatar") for row in avatars[:35]])
            if not image or sys.getsizeof(image.fp) > ctx.guild.filesize_limit:
                await ctx.neutral(
                    (
                        f"Click [**here**](https://{self.bot.config['domain']}/avatars/{user.id}) to view"
                        f" {functions.plural(avatars, bold=True):of your avatar}"
                        if user == ctx.author
                        else (
                            f"Click [**here**](https://{self.bot.config['domain']}/avatars/{user.id}) to view"
                            f" {functions.plural(avatars, bold=True):avatar} of **{user}**"
                        )
                    ),
                    emoji="🖼️",
                )
            else:
                embed = discord.Embed(
                    title="Avatar History",
                    description=(
                        f"Showing `{len(avatars[:35])}` of up to `{len(avatars)}` {'changes' if len(avatars) != 1 else 'change'}\n> For the full list"
                        f" including GIFs click [**HERE**](https://{self.bot.config['domain']}/avatars/{user.id})"
                    ),
                )
                embed.set_author(
                    name=f"{user} ({user.id})",
                    icon_url=user.display_avatar.url,
                )

                embed.set_image(
                    url="attachment://collage.png",
                )
                await ctx.send(
                    embed=embed,
                    file=image,
                )

    # @commands.command(
    #     name="covid",
    #     usage="<country>",
    #     example="USA",
    #     aliases=["corona", "coronavirus", "covid19", "covid-19"],
    # )
    # async def covid(self, ctx: wock.Context, *, country: str = "USA"):
    #     """View COVID-19 statistics for a country"""

    #     response = await self.bot.session.get(
    #         "https://disease.sh/v3/covid-19/countries/" + functions.format_uri(country)
    #     )
    #     data = await response.json()

    #     if data.get("message") == "Country not found or doesn't have any cases":
    #         return await ctx.warn(f"Could find any **COVID-19** data for `{country}`")

    #     embed = discord.Embed(
    #         title=f"COVID-19 Cases - {data['country']}",
    #         description=f"> Information from [**disease.sh**](https://disease.sh)",
    #     )

    #     embed.add_field(
    #         name="Confirmed",
    #         value=humanize.comma(data["cases"]),
    #     )
    #     embed.add_field(
    #         name="Deaths",
    #         value=humanize.comma(data["deaths"]),
    #     )
    #     embed.add_field(
    #         name="Recovered",
    #         value=humanize.comma(data["recovered"]),
    #     )
    #     embed.add_field(
    #         name="Active Cases",
    #         value=humanize.comma(data["active"]),
    #     )
    #     embed.add_field(
    #         name="Critical Cases",
    #         value=humanize.comma(data["critical"]),
    #     )
    #     embed.add_field(
    #         name="Population",
    #         value=humanize.comma(data["population"]),
    #     )
    #     await ctx.send(embed=embed)

    @commands.command(
        name="createembed",
        usage="(embed script)",
        example="{title: wow!}",
        aliases=["embed", "ce"],
    )
    async def createembed(self, ctx: wock.Context, *, script: wock.EmbedScriptValidator):
        """Send an embed to the channel"""

        await script.send(
            ctx,
            bot=self.bot,
            guild=ctx.guild,
            channel=ctx.channel,
            user=ctx.author,
        )

    @commands.command(name="copyembed", usage="(message)", example="dscord.com/chnls/999/..", aliases=["embedcode", "ec"])
    async def copyembed(self, ctx: wock.Context, message: discord.Message):
        """Copy embed code for a message"""

        result = []
        if content := message.content:
            result.append(f"{{content: {content}}}")

        for embed in message.embeds:
            result.append("{embed}")
            if color := embed.color:
                result.append(f"{{color: {color}}}")

            if author := embed.author:
                _author = []
                if name := author.name:
                    _author.append(name)
                if icon_url := author.icon_url:
                    _author.append(icon_url)
                if url := author.url:
                    _author.append(url)

                result.append(f"{{author: {' && '.join(_author)}}}")

            if url := embed.url:
                result.append(f"{{url: {url}}}")

            if title := embed.title:
                result.append(f"{{title: {title}}}")

            if description := embed.description:
                result.append(f"{{description: {description}}}")

            for field in embed.fields:
                result.append(f"{{field: {field.name} && {field.value} && {str(field.inline).lower()}}}")

            if thumbnail := embed.thumbnail:
                result.append(f"{{thumbnail: {thumbnail.url}}}")

            if image := embed.image:
                result.append(f"{{image: {image.url}}}")

            if footer := embed.footer:
                _footer = []
                if text := footer.text:
                    _footer.append(text)
                if icon_url := footer.icon_url:
                    _footer.append(icon_url)

                result.append(f"{{footer: {' && '.join(_footer)}}}")

            if timestamp := embed.timestamp:
                result.append(f"{{timestamp: {str(timestamp)}}}")

        if not result:
            await ctx.warn(f"Message [`{message.id}`]({message.jump_url}) doesn't contain an embed")
        else:
            result = "\n".join(result)
            await ctx.approve(f"Copied the **embed code**\n```{result}```")

    @commands.command(
        name="snipe",
        usage="<index>",
        example="4",
        aliases=["sn", "s"],
    )
    async def snipe(self, ctx: wock.Context, index: int = 1):
        """View deleted messages"""

        messages = await self.bot.redis.lget(f"deleted_messages:{functions.hash(ctx.channel.id)}")
        if not messages:
            return await ctx.warn("No **deleted messages** found in this channel")

        if index > len(messages):
            return await ctx.warn(f"Couldn't find a deleted message at index `{index}`")
        else:
            message = list(
                sorted(
                    messages,
                    key=lambda m: m.get("timestamp"),
                    reverse=True,
                )
            )[index - 1]

        embed = discord.Embed(
            description=(message.get("content") or ("__Message contained an embed__" if message.get("embeds") else "")),
            timestamp=datetime.fromtimestamp(message.get("timestamp")),
        )
        embed.set_author(
            name=message["author"].get("name"),
            icon_url=message["author"].get("avatar"),
        )

        if message.get("attachments"):
            embed.set_image(url=message["attachments"][0])
        elif message.get("stickers"):
            embed.set_image(url=message["stickers"][0])
        embed.set_footer(
            text=f"{index:,} of {functions.plural(messages):message}",
            icon_url=ctx.author.display_avatar,
        )

        await ctx.reply(
            embeds=[
                embed,
                *[discord.Embed.from_dict(embed) for embed in message.get("embeds", [])],
            ]
        )

    @commands.command(
        name="editsnipe",
        usage="<index>",
        example="4",
        aliases=["esnipe", "es", "eh"],
    )
    async def editsnipe(self, ctx: wock.Context, index: int = 1):
        """View edited messages"""

        messages = await self.bot.redis.lget(f"edited_messages:{functions.hash(ctx.channel.id)}")
        if not messages:
            return await ctx.warn("No **edited messages** found in this channel")

        if index > len(messages):
            return await ctx.warn(f"Couldn't find an edited message at index `{index}`")
        else:
            message = list(
                sorted(
                    messages,
                    key=lambda m: m.get("timestamp"),
                    reverse=True,
                )
            )[index - 1]

        embed = discord.Embed(
            description=(message.get("content") or ("__Message contained an embed__" if message.get("embeds") else "")),
            timestamp=datetime.fromtimestamp(message.get("timestamp")),
        )
        embed.set_author(
            name=message["author"].get("name"),
            icon_url=message["author"].get("avatar"),
        )

        if message.get("attachments"):
            embed.set_image(url=message["attachments"][0])
        elif message.get("stickers"):
            embed.set_image(url=message["stickers"][0])
        embed.set_footer(
            text=f"{index:,} of {functions.plural(messages):message}",
            icon_url=ctx.author.display_avatar,
        )

        await ctx.reply(
            embeds=[
                embed,
                *[discord.Embed.from_dict(embed) for embed in message.get("embeds", [])],
            ]
        )

    @commands.command(
        name="reactionsnipe",
        usage="<message>",
        example="dscord.com/chnls/999/..",
        aliases=["rsnipe", "rs", "rh"],
    )
    async def reactionsnipe(self, ctx: wock.Context, *, message: discord.Message = None):
        """View removed reactions"""

        reactions = await self.bot.redis.lget(f"removed_reactions:{functions.hash(ctx.channel.id)}")

        if not reactions:
            return await ctx.warn("No **removed reactions** found in this channel")

        if not message:
            reaction = reactions[0]
            message = ctx.channel.get_partial_message(reaction["message"])
        else:
            reaction = next(
                (reaction for reaction in reactions if reaction["message"] == message.id),
                None,
            )
            if not reaction:
                return await ctx.warn("No **removed reactions** found for that message")

        try:
            await ctx.neutral(
                f"**{self.bot.get_user(reaction.get('user')) or reaction.get('user')}** removed **{reaction.get('emoji')}**"
                f" {discord.utils.format_dt(datetime.fromtimestamp(reaction.get('timestamp')), style='R')}",
                reference=message,
            )
        except discord.HTTPException:
            await ctx.neutral(
                f"**{self.bot.get_user(reaction.get('user')) or reaction.get('user')}** removed **{reaction.get('emoji')}**"
                f" {discord.utils.format_dt(datetime.fromtimestamp(reaction.get('timestamp')), style='R')}"
            )

    @commands.command(
        name="clearsnipes",
        aliases=["clearsnipe", "cs"],
    )
    @commands.has_permissions(manage_messages=True)
    async def clearsnipes(self, ctx: wock.Context):
        """Clear deleted messages from the cache"""

        await self.bot.redis.delete(
            f"deleted_messages:{functions.hash(ctx.channel.id)}",
            f"edited_messages:{functions.hash(ctx.channel.id)}",
            f"removed_reactions:{functions.hash(ctx.channel.id)}",
        )
        await ctx.react_check()

    @commands.group(
        name="highlight",
        usage="(subcommand) <args>",
        example="add rx",
        aliases=["snitch", "hl"],
        invoke_without_command=True,
    )
    async def highlight(self, ctx: wock.Context):
        """Notifies you when a keyword is said"""

        await ctx.send_help()

    @highlight.command(
        name="add",
        usage="(word)",
        example="rx",
        parameters={
            "strict": {
                "require_value": False,
                "description": "Whether the message should be a strict match",
            },
        },
        aliases=["create"],
    )
    @checks.require_dm()
    async def highlight_add(self, ctx: wock.Context, *, word: str):
        """Add a keyword to notify you about"""

        word = word.lower()

        if discord.utils.escape_mentions(word) != word:
            return await ctx.warn("Your keyword can't contain mentions")
        elif len(word) < 2:
            return await ctx.warn("Your keyword must be at least **2 characters** long")
        elif len(word) > 32:
            return await ctx.warn("Your keyword can't be longer than **32 characters**")

        try:
            await self.bot.db.execute(
                "INSERT INTO highlight_words (user_id, word, strict) VALUES ($1, $2, $3)",
                ctx.author.id,
                word,
                ctx.parameters.get("strict"),
            )
        except:
            await ctx.warn(f"You're already being notified about `{word}`")
        else:
            await ctx.approve(f"You'll now be notified about `{word}` " + ("(strict)" if ctx.parameters.get("strict") else ""))

    @highlight.command(
        name="remove",
        usage="(word)",
        example="rx",
        aliases=["delete", "del", "rm"],
    )
    async def highlight_remove(self, ctx: wock.Context, *, word: str):
        """Remove a highlighted keyword"""

        word = word.lower()

        try:
            await self.bot.db.execute(
                "DELETE FROM highlight_words WHERE user_id = $1 AND word = $2",
                ctx.author.id,
                word,
                raise_exceptions=True,
            )
        except:
            await ctx.warn(f"You're not being notified about `{word}`")
        else:
            await ctx.approve(f"You won't be notified about `{word}` anymore")

    @highlight.group(
        name="ignore",
        usage="(user or channel)",
        example="rx#1337",
        aliases=["block"],
        invoke_without_command=True,
    )
    async def highlight_ignore(self, ctx: wock.Context, *, entity: wock.Member | wock.User | discord.TextChannel | discord.CategoryChannel):
        """Ignore a user or channel"""

        if entity.id == ctx.author.id:
            return await ctx.warn("You can't ignore yourself")

        try:
            await self.bot.db.execute(
                "INSERT INTO highlight_block (user_id, entity_id) VALUES ($1, $2)",
                ctx.author.id,
                entity.id,
            )
        except:
            await ctx.warn(
                "You're already ignoring"
                f" [**{entity}**]({entity.jump_url if (isinstance(entity, discord.TextChannel) or isinstance(entity, discord.CategoryChannel)) else ''})"
            )
        else:
            await ctx.approve(
                "Now ignoring"
                f" [**{entity}**]({entity.jump_url if (isinstance(entity, discord.TextChannel) or isinstance(entity, discord.CategoryChannel)) else ''})"
            )

    @highlight_ignore.command(
        name="list",
        aliases=["show", "all"],
    )
    async def highlight_ignore_list(self, ctx: wock.Context):
        """View all ignored users and channels"""

        entities = [
            f"[**{entity}**]({entity.jump_url if (isinstance(entity, discord.TextChannel) or isinstance(entity, discord.CategoryChannel)) else ''})"
            f" (`{entity.id}`)"
            async for row in self.bot.db.fetchiter(
                "SELECT entity_id FROM highlight_block WHERE user_id = $1",
                ctx.author.id,
            )
            if (entity := self.bot.get_user(row.get("entity_id")) or self.bot.get_channel(row.get("entity_id")))
        ]
        if not entities:
            return await ctx.warn("You're not **ignoring** anyone")

        await ctx.paginate(discord.Embed(title="Ignored Entities", description=entities))

    @highlight.command(
        name="unignore",
        usage="(user or channel)",
        example="rx#1337",
        aliases=["unblock"],
    )
    async def highlight_unignore(self, ctx: wock.Context, *, entity: wock.Member | wock.User | discord.TextChannel | discord.CategoryChannel):
        """Unignore a user or channel"""

        try:
            await self.bot.db.execute(
                "DELETE FROM highlight_block WHERE user_id = $1 AND entity_id = $2",
                ctx.author.id,
                entity.id,
            )
        except:
            await ctx.warn(
                "You're not ignoring"
                f" [**{entity}**]({entity.jump_url if (isinstance(entity, discord.TextChannel) or isinstance(entity, discord.CategoryChannel)) else ''})"
            )
        else:
            await ctx.approve(
                "No longer ignoring"
                f" [**{entity}**]({entity.jump_url if (isinstance(entity, discord.TextChannel) or isinstance(entity, discord.CategoryChannel)) else ''})"
            )

    @highlight.command(
        name="list",
        aliases=["show", "all"],
    )
    async def highlight_list(self, ctx: wock.Context):
        """View all highlighted keywords"""

        words = [
            f"**{row['word']}** (strict: {'yes' if row.get('strict') else 'no'})"
            async for row in self.bot.db.fetchiter(
                "SELECT word, strict FROM highlight_words WHERE user_id = $1 ORDER BY length(word) DESC", ctx.author.id
            )
        ]
        if not words:
            return await ctx.warn("You're not being notified about any **keywords**")

        await ctx.paginate(
            discord.Embed(
                title="Highlights",
                description=words,
            )
        )

    @commands.command(
        name="deafen",
        aliases=["deaf", "mid"],
    )
    async def deafen(self, ctx: wock.Context):
        """Server deafen yourself"""

        if not ctx.author.voice:
            return await ctx.warn("You're not in a voice channel")

        await ctx.author.edit(deafen=not ctx.author.voice.deaf)
        await ctx.check()


async def setup(bot: wock.wockSuper):
    await bot.add_cog(miscellaneous(bot))
