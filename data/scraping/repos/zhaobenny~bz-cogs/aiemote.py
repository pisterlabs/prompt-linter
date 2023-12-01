import asyncio
import logging
import random
import re
from typing import Optional

import discord
import tiktoken
from emoji import EMOJI_DATA
from openai import AsyncOpenAI
from redbot.core import Config, checks, commands
from redbot.core.bot import Red
from redbot.core.utils.menus import start_adding_reactions
from redbot.core.utils.predicates import ReactionPredicate
from redbot.core.utils.views import SimpleMenu

logger = logging.getLogger("red.bz_cogs.aiemote")


class AIEmote(commands.Cog):
    MATCH_DISCORD_EMOJI_REGEX = r"<a?:[A-Za-z0-9]+:[0-9]+>"

    def __init__(self, bot):
        super().__init__()
        self.bot: Red = bot
        self.config = Config.get_conf(self, identifier=754069)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.aclient = None

        default_global = {
            "percent": 50,
            "global_emojis": [
                {
                    "description": "A happy face",
                    "emoji": "😀",
                },
                {
                    "description": "A sad face",
                    "emoji": "😢",
                },
            ],
            "extra_instruction": "",
            "optin": [],
            "optout": []
        }

        default_guild = {
            "server_emojis": [],
            "whitelist": [],
            "optin_by_default": False
        }

        self.config.register_guild(**default_guild)
        self.config.register_global(**default_global)

    async def cog_load(self):
        self.whitelist = {}
        all_config = await self.config.all_guilds()
        self.percent = await self.config.percent()
        self.optin_users = await self.config.optin()
        self.optout_users = await self.config.optout()
        for guild_id, config in all_config.items():
            self.whitelist[guild_id] = config["whitelist"]

    @commands.Cog.listener()
    async def on_red_api_tokens_update(self, service_name, api_tokens):
        if service_name == "openai":
            self.initalize_openai()

    async def initalize_openai(self, ctx: commands.Context = None):
        key = (await self.bot.get_shared_api_tokens("openai")).get("api_key")
        if not key and ctx:
            return await ctx.send(
                f"OpenAI API key not set for `aiemote`. "
                f"Please set it with `{ctx.clean_prefix}set api openai api_key,API_KEY`")
        if not key:
            logger.error(F"OpenAI API key not set for `aiemote` yet! Please set it with `{ctx.clean_prefix}set api openai api_key,API_KEY`")
            return

        self.aclient = AsyncOpenAI(
                api_key=key,
                timeout=50.0
        )

    @commands.Cog.listener()
    async def on_message_without_command(self, message: discord.Message):
        ctx: commands.Context = await self.bot.get_context(message)
        if not (await self.is_valid_to_react(ctx)):
            return
        if (self.percent < random.randint(0, 99)):
            return
        emoji = await self.pick_emoji(message)
        if emoji:
            await message.add_reaction(emoji)

    async def pick_emoji(self, message: discord.Message):
        options = "\n"
        emojis = await self.config.guild(message.guild).emojis() or []
        emojis += await self.config.global_emojis() or []
        for index, value in enumerate(emojis):
            options += f"{index}. {value['description']}\n"

        logit_bias = {}
        for i in range(len(emojis)):
            encoded_value = self.encoding.encode(str(i))
            if len(encoded_value) == 1:
                logit_bias[encoded_value[0]] = 100

        system_prompt = f"You are in a chat room. You will pick an emoji for the following message. {await self.config.extra_instruction()} Here are your options: {options} Your answer will be a int between 0 and {len(emojis)-1}."
        content = f"{message.author.display_name} : {self.stringify_any_mentions(message)}"
        try:
            response = await self.aclient.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=1,
                logit_bias=logit_bias,
            )
        except:
            logger.warning("Skipping react! Failed to get response from OpenAI")
            return None

        response = response.choices[0].message.content
        if response.isnumeric():
            index = int(response)
            if index < 0 or index >= len(emojis):
                return None
            partial_emoji = discord.PartialEmoji.from_str(emojis[index]["emoji"])
            return partial_emoji
        else:
            logger.warning(
                f"Skipping react! Non-numeric response from OpenAI: {response}. (Please report to dev if this occurs often)")
            return None

    async def is_valid_to_react(self, ctx: commands.Context):
        if ctx.guild is None or ctx.author.bot:
            return False

        whitelist = self.whitelist.get(ctx.guild.id, [])
        if await self.bot.cog_disabled_in_guild(self, ctx.guild):
            return False

        if (not isinstance(ctx.channel, discord.Thread) and (ctx.channel.id not in whitelist)):
            return False
        if (isinstance(ctx.channel, discord.Thread) and (ctx.channel.parent_id not in whitelist)):
            return False

        if not await self.bot.ignored_channel_or_guild(ctx):
            return False
        if not await self.bot.allowed_by_whitelist_blacklist(ctx.author):
            return False
        if ctx.author.id in self.optout_users:
            return False
        if (not ctx.author.id in self.optin_users) and (not (await self.config.guild(ctx.guild).optin_by_default())):
            return False

        if not self.aclient:
            await self.initalize_openai(ctx)
            if not self.aclient:
                return False

        # skipping images / embeds
        if not ctx.message.content or (ctx.message.attachments and len(ctx.message.attachments) > 0):
            return False

        # skipping long / short messages
        if len(ctx.message.content) > 1500 or len(ctx.message.content) < 10:
            logger.debug(f"Skipping message in {ctx.guild.name} with length {len(ctx.message.content)}")
            return False

        return True

    def stringify_any_mentions(self, message: discord.Message) -> str:
        """
        Converts mentions to text
        """
        content = message.content
        mentions = message.mentions + message.role_mentions + message.channel_mentions

        if not mentions:
            return content

        for mentioned in mentions:
            if mentioned in message.channel_mentions:
                content = content.replace(mentioned.mention, f'#{mentioned.name}')
            elif mentioned in message.role_mentions:
                content = content.replace(mentioned.mention, f'@{mentioned.name}')
            else:
                content = content.replace(mentioned.mention, f'@{mentioned.display_name}')

        return content

    @commands.group(name="aiemote", alias=["ai_emote"])
    @checks.admin_or_permissions(manage_guild=True)
    async def aiemote(self, _):
        """ Totally not glorified sentiment analysis™

            Picks a reaction for a message using gpt-3.5-turbo

            To get started, please add a channel to the whitelist with:
            `[p]aiemote allow <#channel>`
        """
        pass

    @aiemote.command(name="whitelist")
    @checks.admin_or_permissions(manage_guild=True)
    async def whitelist_list(self, ctx: commands.Context):
        """ List all channels in the whitelist """
        whitelist = self.whitelist.get(ctx.guild.id, [])
        if not whitelist:
            return await ctx.send("No channels in whitelist")
        channels = [ctx.guild.get_channel(channel_id) for channel_id in whitelist]
        embed = discord.Embed(title="Whitelist", color=await ctx.embed_color())
        embed.add_field(name="Channels", value="\n".join([channel.mention for channel in channels]))
        await ctx.send(embed=embed)

    @aiemote.command(name="allow", aliases=["add"])
    @checks.admin_or_permissions(manage_guild=True)
    async def whitelist_add(self, ctx: commands.Context, channel: discord.TextChannel):
        """ Add a channel to the whitelist

            *Arguments*
            - `<channel>` The mention of channel
        """
        whitelist = self.whitelist.get(ctx.guild.id, [])
        if channel.id in whitelist:
            return await ctx.send("Channel already in whitelist")
        whitelist.append(channel.id)
        self.whitelist[ctx.guild.id] = whitelist
        await self.config.guild(ctx.guild).whitelist.set(whitelist)
        return await ctx.tick()

    @aiemote.command(name="remove", aliases=["rm"])
    @checks.admin_or_permissions(manage_guild=True)
    async def whitelist_remove(self, ctx: commands.Context, channel: discord.TextChannel):
        """ Remove a channel from the whitelist

            *Arguments*
            - `<channel>` The mention of channel
        """
        whitelist = self.whitelist.get(ctx.guild.id, [])
        if channel.id not in whitelist:
            return await ctx.send("Channel not in whitelist")
        whitelist.remove(channel.id)
        self.whitelist[ctx.guild.id] = whitelist
        await self.config.guild(ctx.guild).whitelist.set(whitelist)
        return await ctx.tick()

    @aiemote.command(name="optinbydefault", alias=["optindefault"])
    @checks.admin_or_permissions(manage_guild=True)
    async def optin_by_default(self, ctx: commands.Context):
        """ Toggles whether users are opted in by default in this server

            This command is disabled for servers with more than 150 members.
        """
        if len(ctx.guild.members) > 150:
            # if you STILL want to enable this for a server with more than 150 members
            # add the line below to the specific guild in the cog's settings.json:
            # "optin_by_default": true
            # insert concern about user privacy and getting user consent here
            return await ctx.send("You cannot enable this setting for servers with more than 150 members.")
        value = not await self.config.guild(ctx.guild).optin_by_default()
        await self.config.guild(ctx.guild).optin_by_default.set(value)
        embed = discord.Embed(
            title="Users are now opted in by default in this server:",
            description=f"{value}",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @aiemote.command(name="optin")
    async def optin_user(self, ctx: commands.Context):
        """ Opt in of sending your message to OpenAI (bot-wide)

            This will allow the bot to react to your messages
        """
        optin = await self.config.optin()
        optout = await self.config.optout()

        if ctx.author.id in await self.config.optin() and ctx.author.id not in self.optout_users:
            return await ctx.send("You are already opted in bot-wide")

        optin.append(ctx.author.id)
        self.optin_users.append(ctx.author.id)
        await self.config.optin.set(optin)

        if ctx.author.id in optout:
            optout.remove(ctx.author.id)
            self.optout_users.remove(ctx.author.id)
            await self.config.optout.set(optout)

        await ctx.send("You are now opted in bot-wide")

    @aiemote.command(name="optout")
    async def optout_user(self, ctx: commands.Context):
        """ Opt out of sending your message to OpenAI (bot-wide)

            The bot will no longer react to your messages
        """
        optin = await self.config.optin()
        optout = await self.config.optout()

        if not ctx.author.id in await self.config.optin() and ctx.author.id in self.optout_users:
            return await ctx.send("You are already opted out")

        if ctx.author.id in optin:
            optin.remove(ctx.author.id)
            self.optin_users.remove(ctx.author.id)
            await self.config.optin.set(optin)

        optout.append(ctx.author.id)
        self.optout_users.append(ctx.author.id)
        await self.config.optout.set(optout)

        await ctx.send("You are now opted out bot-wide")

    @commands.group(name="aiemoteowner", alias=["ai_emote_admin"])
    @checks.is_owner()
    async def aiemote_owner(self, _):
        """ Owner only commands for aiemote
        """
        pass

    @aiemote_owner.command(name="instruction", aliases=["extra_instruction", "extra"])
    @checks.is_owner()
    async def set_extra_instruction(self, ctx: commands.Context, *, instruction: Optional[str]):
        """ Add additonal (prompting) instruction for the langauge model when picking an emoji

            *Arguments*
            - `<instruction>` The extra instruction to use
        """
        if not instruction:
            await self.config.extra_instruction.set("")
        else:
            await self.config.extra_instruction.set(instruction)
        return await ctx.tick()

    async def check_valid_emoji(self, ctx: commands.Context, emoji):
        if emoji in EMOJI_DATA.keys():
            return True
        if (not bool(re.fullmatch(self.MATCH_DISCORD_EMOJI_REGEX, emoji))):
            await ctx.send("Invalid emoji!")
            return False
        emoji = discord.PartialEmoji.from_str(emoji)
        isBotEmoji = bool(discord.utils.get(self.bot.emojis, name=emoji.name, id=emoji.id))
        if not isBotEmoji:
            await ctx.send("Invalid emoji! Custom emojis must be usable by the bot itself")
            return False
        return True

    async def add_emoji(self, ctx: commands.Context, emoji_list, emoji, description):
        if any(item["emoji"] == str(emoji) for item in emoji_list):
            await ctx.send("Emoji already in list")
            return False

        emoji_list.append({
            "description": description,
            "emoji": str(emoji)
        })
        return True

    async def remove_emoji(self, ctx: commands.Context, emoji_list, emoji):
        index = next((i for i, item in enumerate(emoji_list) if item["emoji"] == str(emoji)), -1)
        if index == -1:
            await ctx.send("Emoji not in list")
            return False

        del emoji_list[index]
        return True

    @aiemote_owner.command(name="add")
    @checks.is_owner()
    async def add_global_emoji(self, ctx: commands.Context, emoji, *, description: str):
        """ Add an emoji to the global list

            *Arguments*
            - `<emoji>` The emoji to add
            - `<description>` A description of the emoji to be used by OpenAI
        """
        if not await self.check_valid_emoji(ctx, emoji):
            return
        emojis = await self.config.global_emojis()
        if not emojis:
            emojis = []
        if await self.add_emoji(ctx, emojis, emoji, description):
            await self.config.global_emojis.set(emojis)
            await ctx.tick()

    @aiemote_owner.command(name="remove", aliases=["rm"])
    @checks.is_owner()
    async def remove_global_emoji(self, ctx: commands.Context, emoji):
        """ Remove an emoji from the global list

            *Arguments*
            - `<emoji>` The emoji to remove
        """
        if not await self.check_valid_emoji(ctx, emoji):
            return

        emojis = await self.config.global_emojis()
        if await self.remove_emoji(ctx, emojis, emoji):
            await self.config.global_emojis.set(emojis)
            await ctx.tick()

    async def create_emoji_embed(self, ctx, title: str, emojis: list):
        embeds = []
        chunk_size = 8

        if len(emojis) == 0:
            embed = discord.Embed(title=title, description="None", color=await ctx.embed_color())
            embeds.append(embed)
            return embeds

        for i in range(0, len(emojis), chunk_size):
            embed = discord.Embed(title=title, color=await ctx.embed_color())

            chunk = emojis[i: i + chunk_size]
            for item in chunk:
                partial_emoji = discord.PartialEmoji.from_str(item["emoji"])
                emoji = str(partial_emoji)
                embed.add_field(name=emoji, value=item["description"], inline=False)

            embeds.append(embed)

        if len(embeds) > 1:
            for i, page in enumerate(embeds):
                page.set_footer(text=f"Page {i+1} of {len(embeds)}")

        return embeds

    @aiemote_owner.command(name="config", aliases=["settings", "list", "conf"])
    @checks.is_owner()
    async def list_all_emoji(self, ctx: commands.Context):
        """ List all emojis in the global list (and current server list)
        """
        emojis = await self.config.global_emojis()
        globalembeds = await self.create_emoji_embed(ctx, "Global Emojis", emojis)
        emojis = await self.config.guild(ctx.guild).server_emojis()
        serverembeds = await self.create_emoji_embed(ctx, "Current Server-specific Emojis", emojis)
        settingsembed = discord.Embed(title="Main Settings", color=await ctx.embed_color())
        settingsembed.add_field(name="Percent Chance", value=f"{self.percent}%", inline=False)
        settingsembed.add_field(name="Additonal Instruction", value=await self.config.extra_instruction() or "None", inline=False)
        await ctx.send(embed=settingsembed)
        if len(globalembeds) > 1:
            await (SimpleMenu(globalembeds)).start(ctx)
        else:
            await ctx.send(embed=globalembeds[0])
        if len(serverembeds) > 1:
            await (SimpleMenu(serverembeds)).start(ctx)
        else:
            await ctx.send(embed=serverembeds[0])

    @aiemote_owner.command(name="reset")
    @checks.is_owner()
    async def reset_all_settings(self, ctx: commands.Context):
        """
        Reset *all* settings
        """
        embed = discord.Embed(
            title="Are you sure?",
            description="This will reset all settings to default! (Including ALL per server lists)",
            color=await ctx.embed_color())
        confirm = await ctx.send(embed=embed)
        start_adding_reactions(confirm, ReactionPredicate.YES_OR_NO_EMOJIS)
        pred = ReactionPredicate.yes_or_no(confirm, ctx.author)
        try:
            await ctx.bot.wait_for("reaction_add", timeout=10.0, check=pred)
        except asyncio.TimeoutError:
            return await confirm.edit(embed=discord.Embed(title="Cancelled.", color=await ctx.embed_color()))
        if pred.result is False:
            return await confirm.edit(embed=discord.Embed(title="Cancelled.", color=await ctx.embed_color()))
        else:
            await self.config.clear_all_guilds()
            await self.config.clear_all_globals()
            self.whitelist = {}
            self.percent = 50
            return await confirm.edit(embed=discord.Embed(title="Cleared.", color=await ctx.embed_color()))

    @aiemote_owner.command(name="sadd")
    @checks.is_owner()
    async def add_server_emoji(self, ctx: commands.Context, emoji, *, description: str):
        """ Add an emoji to this current server list

            *Arguments*
            - `<emoji>` The emoji to add
            - `<description>` A description of the emoji to be used by OpenAI
        """
        if not await self.check_valid_emoji(ctx, emoji):
            return
        emojis = await self.config.guild(ctx.guild).server_emojis()
        if not emojis:
            emojis = []
        if await self.add_emoji(ctx, emojis, emoji, description):
            await self.config.guild(ctx.guild).server_emojis.set(emojis)
            await ctx.tick()

    @aiemote_owner.command(name="sremove", aliases=["srm"])
    @checks.is_owner()
    async def remove_server_emoji(self, ctx: commands.Context, emoji):
        """ Remove an emoji from this current server list

            *Arguments*
            - `<emoji>` The emoji to remove
        """
        if not await self.check_valid_emoji(ctx, emoji):
            return

        emojis = await self.config.guild(ctx.guild).server_emojis()
        if await self.remove_emoji(ctx, emojis, emoji):
            await self.config.guild(ctx.guild).server_emojis.set(emojis)
            await ctx.tick()

    @aiemote_owner.command(name="percent")
    @checks.is_owner()
    async def set_percent(self, ctx: commands.Context, percent: int):
        """ Set the chance that the bot will react to a message (for all servers bot is in)

            *Arguments*
            - `<percent>` The percent chance that the bot will react to a message
        """
        if percent < 0 or percent > 100:
            return await ctx.send("Invalid percent")
        self.percent = percent
        await self.config.percent.set(percent)
        return await ctx.tick()
