import os
import sys
import yaml
import nextcord
from typing import Optional
from nextcord.ext import commands
from nextcord import Interaction, SlashOption, ChannelType
from nextcord.abc import GuildChannel
from noncommands import summarizer
import openai
from noncommands.chatsplit import chatsplit


with open("config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Here we name the cog and create a new class for the cog.
class TLDR(commands.Cog, name="tldr"):
    def __init__(self, bot):
        self.bot = bot

    @nextcord.slash_command(name="tldrchannel", description="Get a TLDR of X number of past messages on the channel.")
    async def tldrchannel(self, interaction: Interaction, number: Optional[int] = SlashOption(description="The number of past messages to summarize", required=True, min_value=2, max_value=200)):
        """
        [NumberOfMessages] Get a TLDR of X number of past messages on the channel.
        """

        await interaction.response.defer()

        messages = await interaction.channel.history(limit=number).flatten()
        
        chats = [{"role": "system", "content": "You are a summarizing machine. You are going to be given a number of messages from a discord server, then you are going to summarize the general conversation that has been happening."}]
        for m in messages:
            chats.append({"role": "user", "content": f"[{m.author.display_name}]: {m.content}"})
        chats.append({"role": "user", "content": "Please summarize the previous messages in a concise way to catch the user up on what has been happening. Make sure to hit the important details and to not include any unnecessary information."})

        chatCompletion = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=chats)
        response = chatCompletion.choices[0].message.content

        messages = chatsplit(response)

        for message in messages:
            await interaction.followup.send(message)
    
    @nextcord.slash_command(name="tldr", description="Get a TLDR of a web page.")
    async def tldr(self, interaction: Interaction, url: Optional[str] = SlashOption(description="The URL of the web page to summarize", required=True)):
        """
        [URL] Get a TLDR a web page.
        """
        try:
            await interaction.response.send_message(embed=summarizer.getSummaryUrl(config, url))
        except:
            await interaction.response.send_message("There's something odd about that link. Either they won't let me read it or you sent it wrongly.")
        


# And then we finally add the cog to the bot so that it can load, unload, reload and use it's content.
def setup(bot):
    bot.add_cog(TLDR(bot))