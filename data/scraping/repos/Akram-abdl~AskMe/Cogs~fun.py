import discord
from discord.ext import commands
import os
import openai
class Fun(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(name="joke")
    async def joke(self, ctx: commands.Context):
        # Use OpenAI to generate a response
        prompt = "Tell me a joke."
        response = openai.Completion.create(
            engine="text-curie-001",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Send the response back to the user
        await ctx.respond(response.choices[0].text.strip())

    @commands.slash_command(name="fact")
    async def fact(self, ctx: commands.Context):
        # Use OpenAI to generate a response
        prompt = "Tell me a fun fact."
        response = openai.Completion.create(
            engine="text-curie-001",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Send the response back to the user
        await ctx.respond(response.choices[0].text.strip())

def setup(bot):
    bot.add_cog(Fun(bot))
