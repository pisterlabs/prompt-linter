from datetime import datetime
import discord
from discord.ext import commands
from discord import app_commands
import config
from helper import on_interaction
import openai
from threading import Thread
from nse import megascrape,scrape
def airesponse(q):
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=q,#"Write Python code for checking if number is prime or not",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']
Cog = commands.Cog



class Miscellaneous(Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
    @Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.channel.id == config.voting:
            await message.add_reaction('\U0001F44D')
            await message.add_reaction('🤷‍♂️')
            await message.add_reaction('\U0001F44E')
    @Cog.listener()
    async def on_message_edit(self, before,message: discord.Message):
        if message.channel.id == config.voting:
            await message.add_reaction('\U0001F44D')
            await message.add_reaction('🤷‍♂️')
            await message.add_reaction('\U0001F44E')
    @app_commands.command()
    @on_interaction
    async def openai(self, interaction: discord.Interaction,question:str):
        if interaction.user.get_role(config.admin):
            await interaction.response.defer(ephemeral=True)
            await interaction.followup.send(content=question+airesponse(question),ephemeral=True)
            return True
  
    @app_commands.command()
    @on_interaction
    async def remove_role(self, interaction: discord.Interaction, role: discord.Role):
        await interaction.response.defer()
        if interaction.user.get_role(config.admin):
            for member in role.members:
                await member.remove_roles(role)
        await interaction.followup.send(f'Removed all members from {role}.')
        return True
    
    @app_commands.command(name="attachment_link",description="Converts attachment to link.")
    @on_interaction
    async def attachment_link(self, interaction: discord.Interaction,attachment: discord.Attachment):
        await interaction.response.send_message(content=attachment.url)
        return False
    @app_commands.command(name="helper", description="To ping helpers.")
    @app_commands.checks.cooldown(1,30*60)
    @on_interaction
    async def helper(self, interaction: discord.Interaction):
        phods = self.bot.get_guild(config.phods)
        await interaction.response.send_message(phods.get_role(config.helper).mention)
        return True
    @app_commands.command(name="resources", description="See resources")
    @on_interaction
    async def resource(self, interaction: discord.Interaction):
        resources = discord.Embed(title="Physics Resources",colour=config.green,description="[A Comprehensive List of Physics Olympiad Resources](https://artofproblemsolving.com/community/c164h2094716_a_comprehensive_list_of_physics_olympiad_resources)\n[Various Textbooks and Solutions](https://discordapp.com/channels/601528888908185655/601884131005169743/707334765673447454)")
        await interaction.response.send_message(embed=resources)
    @app_commands.command(name="help", description="To know about the commands.")
    @on_interaction
    async def help(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Physbot3.0",colour=config.green,description="A multipurpose bot made for phods.")
        embed.add_field(name="Miscellaneous",value="/help -> to bring this embed\n/helper -> to ping @helper\n/resources -> to know some resources for physoly.\n/attachement_link -> Converts attachment into a link.",inline=False)
        embed.add_field(name="PoTD",value="/potd fetch <num> -> Bring the potd of that number\n/potd solution <num> -> Bring the solution of that potd of that number\n/potd submit <num> <attachment> -> to submit your soln of the live potd. (If you want to get on the leaderboard)\n/potd upload -> request your potd to upload.",inline=False)
        embed.add_field(name="QoTD",value="/qotd fetch <num> -> Bring the qotd of that number\n/qotd solution <num> -> Bring the solution of that qotd of that number\n/qotd submit <num> <answer> -> to submit your soln of the live qotd. (If you want to get on the leaderboard)",inline=False)
        await interaction.response.send_message(embed=embed,ephemeral=True)
        return True
    @helper.error
    async def on_submit_error(self,interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.CommandOnCooldown):
            await interaction.response.send_message(str(error), ephemeral=True)
    async def cog_app_command_error(self, interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
        await self.bot.get_channel(config.log2).send((await self.bot.fetch_user(config.proelectro)).mention,embed=discord.Embed(color=config.red,title=str(interaction.user),description=str(error)))

async def setup(bot):
    await bot.add_cog(Miscellaneous(bot))
