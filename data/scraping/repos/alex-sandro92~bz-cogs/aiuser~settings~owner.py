import logging
from typing import Optional

import discord
import openai
from redbot.core import checks, commands

from aiuser.abc import MixinMeta

logger = logging.getLogger("red.bz_cogs.aiuser")


class OwnerSettings(MixinMeta):
    @commands.group(aliases=["ai_userowner"])
    @checks.is_owner()
    async def aiuserowner(self, _):
        """ For some settings that apply bot-wide."""
        pass

    @aiuserowner.command(name="maxpromptlength")
    async def max_prompt_length(self, ctx: commands.Context, length: int):
        """ Sets the maximum character length of a prompt that can set by admins in any server. """
        if length < 1:
            return await ctx.send("Please enter a positive integer.")
        await self.config.max_prompt_length.set(length)
        embed = discord.Embed(
            title="The maximum prompt length is now:",
            description=f"{length}",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @aiuserowner.command(name="maxtopiclength")
    async def max_topic_length(self, ctx: commands.Context, length: int):
        """ Sets the maximum character length of a topic that can set by any server. """
        if length < 1:
            return await ctx.send("Please enter a positive integer.")
        await self.config.max_topic_length.set(length)
        embed = discord.Embed(
            title="The maximum topic length is now:",
            description=f"{length}",
            color=await ctx.embed_color())
        return await ctx.send(embed=embed)

    @aiuserowner.command()
    async def endpoint(self, ctx: commands.Context, url: Optional[str]):
        """ Sets the OpenAI endpoint to a custom one (must be OpenAI API compatible)

            Reset to official OpenAI endpoint with `[p]aiuseradmin endpoint clear`
        """
        if not url or url in ["clear", "reset"]:
            openai.api_base = "https://api.openai.com/v1"
            await self.config.custom_openai_endpoint.set(None)
        else:
            await self.config.custom_openai_endpoint.set(url)
            openai.api_base = url

        embed = discord.Embed(title="Bot Custom OpenAI endpoint", color=await ctx.embed_color())
        embed.add_field(
            name=":warning: Warning :warning:", value="All model/parameters selections for each server may need changing.", inline=False)

        if url:
            embed.description = f"Endpoint set to {url}."
            embed.add_field(
                name="Models", value="Third party models may have undesirable results, compared to OpenAI.", inline=False)
            embed.add_field(
                name="Note", value="This is an advanced feature. \
                    \n If you don't know what you're doing, don't use it",  inline=False)
        else:
            embed.description = "Endpoint reset back to offical OpenAI endpoint."

        await ctx.send(embed=embed)
