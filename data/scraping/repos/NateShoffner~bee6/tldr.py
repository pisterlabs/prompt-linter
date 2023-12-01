import discord
import openai
from discord.ext import commands

class Summarize(commands.Cog):
    
    def __init__(self, bot, config, logger):
        self.bot = bot
        self.config = config
        self.cog_config = config['cogs']['tldr']
        self.logger = logger

    @commands.Cog.listener()
    async def on_ready(self):
        self.logger.info('Summurize ready')

    @commands.command()
    async def tldr(self, ctx: commands.Context):

        if ctx.message.type != discord.MessageType.reply:
            return

        await ctx.message.channel.typing()

        referenced_message = await ctx.message.channel.fetch_message(ctx.message.reference.message_id)

        self.logger.info("Generating Summary...")
        openai.api_key = self.cog_config['api_key']
        response = openai.Completion.create(engine="davinci",prompt=referenced_message.content,temperature=0.3,
                    max_tokens=140,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

        summary = response["choices"][0]['text']
        await ctx.message.channel.send("Requesed Summary:\n```" + summary + '```')