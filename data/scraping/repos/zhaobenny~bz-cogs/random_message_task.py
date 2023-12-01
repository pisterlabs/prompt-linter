import datetime
import logging
import random

import openai
from discord.ext import tasks

from aiuser.abc import MixinMeta
from aiuser.common.utilities import format_variables
from aiuser.messages_list.messages import create_messages_list
from aiuser.response.chat.openai import OpenAI_Chat_Generator
from aiuser.response.chat.response import ChatResponse

logger = logging.getLogger("red.bz_cogs.aiuser")


class RandomMessageTask(MixinMeta):
    @tasks.loop(minutes=33)
    async def random_message_trigger(self):
        if not self.openai_client:
            return
        if not self.bot.is_ready():
            return
        try:
            for guild_id, channels in self.channels_whitelist.items():
                guild = self.bot.get_guild(guild_id)
                if not guild or await self.bot.cog_disabled_in_guild(self, guild):
                    continue
                if not await self.config.guild(guild).random_messages_enabled():
                    continue
                if random.random() > await self.config.guild(guild).random_messages_percent():
                    continue

                if not channels:
                    continue

                channel = guild.get_channel(channels[random.randint(0, len(channels) - 1)])

                if not channel:
                    continue

                try:
                    last = await channel.fetch_message(channel.last_message_id)
                except:
                    continue

                if last.author.id == guild.me.id:
                    # skip spamming channel with random event messages
                    continue

                last_created = last.created_at.replace(tzinfo=datetime.timezone.utc)

                if (abs((datetime.datetime.now(datetime.timezone.utc) - last_created).total_seconds())) < 3600:
                    # only sent to channels with 1 hour since last message
                    continue

                ctx = await self.bot.get_context(last)

                if not (await self.bot.ignored_channel_or_guild(last)):
                    continue

                logger.debug(f"Sending random message to #{channel.name} at {guild.name}")
                prompt = await self.config.channel(last.channel).custom_text_prompt() or await self.config.guild(last.guild).custom_text_prompt()
                message_list = await create_messages_list(self, ctx, prompt=prompt)
                topics = await self.config.guild(guild).random_messages_topics() or ["nothing"]
                topic = format_variables(ctx, topics[random.randint(0, len(topics) - 1)])
                await message_list.add_system(f"You are not responding to a message. Do not greet anyone. You are to start a conversation about the following: {topic}", index=len(message_list) + 1)
                chat = OpenAI_Chat_Generator(self, ctx, message_list)
                response = ChatResponse(ctx, self.config, chat)
                return await response.send(standalone=True)

        except:
            logger.error(f"Exception in random message task", exc_info=True)
