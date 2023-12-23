from twitchio.ext import commands, routines
import logging
import re
import asyncio
from OpenAI import OpenAI
from Config import Config
from Utilities import Utilities
from Dataclasses import ChatLevel


class TwitchBot(commands.Bot):
    def __init__(
        self,
        access_token,
        client_id,
        client_secret,
        bot_name,
        *args,
        **kwargs,
    ):
        super().__init__(
            token=access_token,
            prefix="!",
            client_id=client_id,
            client_secret=client_secret,
            case_insensitive=True,
        )
        logging.debug(f"Access token: {access_token}")
        self._bot_name = bot_name if bot_name is not None else "botdelicious"
        self._pattern_direct = rf"@?{bot_name}[:;,. ]"
        self._pattern_indirect = rf".*{bot_name}.*"
        self.ai_instances = {}
        self.active_channels = []
        logging.info("TwitchBot initialized")

    async def event_ready(self):
        self.routine_check.start(stop_on_error=False)
        self.rejoin_channels.start(stop_on_error=False)
        logging.info(f"Ready | {self.nick}")

    async def event_channel_joined(self, channel):
        logging.info(f"Join event! Channel:{channel}")
        if channel.name not in self.active_channels:
            await self.send_message_to_channel(channel.name, f"Hello, world!")
        self.active_channels.append(channel.name)
        self.ai_instances[channel.name] = OpenAI(
            Config.get(channel.name, "org"),
            Config.get(channel.name, "key"),
            Config.get(channel.name, "prompt_message"),
            Config.get(channel.name, "thinking_message"),
            Config.get(channel.name, "error_message"),
            Config.get(channel.name, "memory_size"),
            Config.get(channel.name, "chat_wide_conversation"),
        )

    async def event_message(self, message):
        logging.info(
            f"{message.channel.name} | {message.author.name if message.author else self._bot_name}:: {message.content}"
        )
        if message.echo:
            return

        _pattern = (
            self._pattern_indirect
            if Config.get(message.channel.name, "all_mentions")
            else self._pattern_direct
        )
        if re.match(_pattern, message.content, re.IGNORECASE):
            logging.debug("Matched")
            logging.info(message.author)
            if self.author_meets_level_requirements(
                message.channel.name, message.author, "chat_level"
            ):
                reply = await self.ai_instances[message.channel.name].chat(
                    username=message.author.name,
                    message=message.content,
                    channel=message.channel.name,
                )
                if reply:
                    await self.send_message_to_channel(
                        message.channel.name, reply
                    )

        if message.content[0] == "!":
            await self.handle_commands(message)
            return

    async def event_token_expired(self):
        logging.info("Token expired")
        return await Utilities.update_twitch_access_token()

    @commands.command()
    async def so(self, ctx: commands.Context):
        if self.author_meets_level_requirements(
            ctx.channel.name, ctx.author, "shoutout_level"
        ):
            username = Utilities.find_username(ctx.message.content)
            target = await self.fetch_user_info(username) if username else None
            logging.debug(f"Target: {target}")
            failed = False if target else username
            shoutout_message = await self.ai_instances[
                ctx.channel.name
            ].shoutout(target=target, author=ctx.author.name, failed=failed)
            if shoutout_message:
                await self.send_message_to_channel(
                    ctx.channel.name, shoutout_message
                )

    @routines.routine(seconds=3, iterations=3)
    async def routine_check(self):
        print("Routine check")
        logging.debug(
            f"Routine check {self.routine_check.completed_iterations + 1} completed,"
            f"{self.routine_check.remaining_iterations - 1} remaining"
        )

    @routine_check.error
    async def routine_check_error(self, error: Exception):
        logging.error(f"Routine check error: {error}")

    @routines.routine(hours=1)
    async def rejoin_channels(self):
        logging.debug("Rejoin channels")
        await self.join_channels(Config.get_twitch_channels())

    async def stop_bot(self):
        logging.info("Stopping bot")
        self.routine_check.cancel()
        await self.close()
        logging.info("Bot stopped")

    async def send_message_to_channel(self, channel, message):
        for attempt in range(3):
            chan = self.get_channel(channel)
            if chan is not None:
                break
            await asyncio.sleep(2)
        else:
            return False

        # Split the message into chunks of up to 500 characters
        message_chunks = []
        while message:
            if len(message) > 500:
                last_space_or_punctuation = re.search(
                    r"[\s\.,;!?-]{1,}[^\s\.,;!?-]*$", message[:500]
                )
                if last_space_or_punctuation:
                    split_at = last_space_or_punctuation.start()
                else:
                    split_at = 500

                chunk = message[:split_at]
                message = message[split_at:].lstrip()
            else:
                chunk = message
                message = ""

            message_chunks.append(chunk)

        # Send each chunk as a separate message
        for chunk in message_chunks:
            self.loop.create_task(chan.send(chunk))
            await asyncio.sleep(2)

    def author_meets_level_requirements(
        self, channel, chatter, type="chat_level"
    ):
        chatter_level = self.translate_chatter_level(chatter)
        type_level = (
            ChatLevel[Config.get(channel, type)]
            if Config.get(channel, type)
            else ChatLevel.VIEWER
        )
        return self.compare_levels(chatter_level, type_level)

    def translate_chatter_level(self, chatter):
        if chatter.is_broadcaster:
            return ChatLevel.BROADCASTER
        if chatter.is_mod:
            return ChatLevel.MODERATOR
        if chatter.is_subscriber:
            return ChatLevel.SUBSCRIBER
        if chatter.is_vip:
            return ChatLevel.VIP
        return ChatLevel.VIEWER

    def compare_levels(self, chatter_level, required_level):
        return chatter_level.value >= required_level.value

    async def fetch_user_info(self, username):
        targets = await self.fetch_users(names=[username])
        if targets and targets[0] is not None:
            name = targets[0].name
            display_name = targets[0].display_name
            description = targets[0].description
            streams = await self.fetch_streams(user_ids=[targets[0].id])
            is_live = True if streams and streams[0] is not None else False
            channels = await self.fetch_channels(
                broadcaster_ids=[targets[0].id]
            )
            game_name = None
            tags = None
            title = None
            if channels and channels[0] is not None:
                channel = channels[0]
                title = channel.title
                game_name = channel.game_name
                tags = channel.tags
            return {
                "name": name,
                "display_name": display_name,
                "description": description,
                "is_live": is_live,
                "game_name": game_name,
                "tags": tags,
                "title": title,
            }
        else:
            return None
