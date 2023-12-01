import asyncio
import importlib
import nextcord
from nextcord.ext import commands
from utils import setup_modules
import openai
from logger import setup_logger, setup_logging_events # something is wrong with the logging. 
from pydactyl import PterodactylClient
from nextcord.ext import commands, application_checks
import os
from dotenv import load_dotenv


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
# Import the setup_botapollo function from botapollo.py

# this is for the apollo server! 
os.environ["PANEL_URL"] = "https://control.sparkedhost.us"
os.environ["PANEL_KEY"] = "ptlc_ZfGef77I1lK5PP7wSBUSLAv6c5cd8bj2yQggADVr0pu"
os.environ["AUTHORIZED_ROLE"] = "1086313112673714339"


bot = commands.Bot()
api = PterodactylClient(os.getenv("PANEL_URL"), os.getenv("PANEL_KEY"))



@bot.event
async def on_application_command_error(interaction, error):
    if isinstance(error, nextcord.ApplicationCheckFailure):
        await interaction.response.send_message(
            ":warning: You are not authorized to use this command!"
        )
    else:
        await interaction.response.send_message(
            f":x: An error occurred while processing this command. \n```py\n{error}\n```"
        )
# end of fucntion for the apollo server

# for the apollo commands

# Creates a new instance of the bot class
intents = nextcord.Intents.all()
intents.guild_messages = True

bot = commands.Bot(command_prefix='!', intents=nextcord.Intents.all())

openai.api_key = "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"



# we have reached the 100 commands on to more basic
# this is how to set up the commands from the folder slashcommands
modules_to_setup = [
    "slash_commands.help",
    "slash_commands.welcome",
    "slash_commands.space",
    "slash_commands.quote",
    "slash_commands.time",
    "slash_commands.roll",
    "slash_commands.flip",
    "slash_commands.template",
    "slash_commands.ping", 
    "slash_commands.info",
    "slash_commands.chat",
    "slash_commands.servers",
    "slash_commands.ticket",
    "slash_commands.rules",
    "slash_commands.warn",
    "slash_commands.clear",
    "slash_commands.lock",
    "slash_commands.unlock",
    "slash_commands.slow",
    "slash_commands.name",
    "slash_commands.role",
    "slash_commands.tempban",
    "slash_commands.tempmute",
    "slash_commands.strike",
    "slash_commands.tracker",
    "slash_commands.ban",
    "slash_commands.kick",
    "slash_commands.unban",
    "slash_commands.mute",
    "slash_commands.unmute",
    "slash_commands.addrole",
    "slash_commands.removerole",
    "slash_commands.mutevoice",
    "slash_commands.def",
    "slash_commands.undef",
    "slash_commands.announce",
    "slash_commands.remind",
    "slash_commands.talk",
    "slash_commands.suggestions", 
    "slash_commands.code",
    "slash_commands.quotetext",
    "slash_commands.encode",
    "slash_commands.password",
    "slash_commands.reverse",
    "slash_commands.wordcount",
    "slash_commands.add",
    "slash_commands.chucknorris",
    "slash_commands.dadjoke",
    "slash_commands.define",
    "slash_commands.progjoke",
    "slash_commands.trivia",
    "slash_commands.lyrics",
    "slash_commands.calculate",
    "slash_commands.gamedeals",
    "slash_commands.advertise2",
    "slash_commands.advertise",
    "slash_commands.celsius",
    "slash_commands.wiki",
    "slash_commands.translate",#
    "slash_commands.summ",
    "slash_commands.gencode",
    "slash_commands.genpoem",
    "slash_commands.genstory",
    "slash_commands.genlyrics",
    "slash_commands.product",#
    "slash_commands.images",
    "slash_commands.advice",
    "slash_commands.genpro",
    "slash_commands.legal",
    "slash_commands.poll",
    "slash_commands.edit",
    "slash_commands.techdoc",
    "slash_commands.verifyrules",
    "slash_commands.goodrules",
    "slash_commands.countdown",
    "slash_commands.news",
    "slash_commands.reactionrules",
    "slash_commands.purge",
    "slash_commands.user",
    "slash_commands.battery",
    "slash_commands.eventer",# make go to another channel
    "slash_commands.blackjack",
    
    # start of langauge model 
    "slash_commands.startlang",
    "slash_commands.practlang",
    "slash_commands.langquiz",
    "slash_commands.scam",
    "slash_commands.quiz",
    "slash_commands.vocab",
    "slash_commands.langpart",
    
    # end of it 
    "slash_commands.sentiment",
    "slash_commands.mach",
    "slash_commands.es",
    "slash_commands.levelup", # end of 100 slash commmands (maybe need to combine some!)
    # Add more modules here as needed
    "slash_commands.tt",
]
setup_modules(bot, modules_to_setup)




# changelog 
with open('changelog.txt', 'r', encoding='utf-8') as f:
    changelog = f.read()

changelog_channel_id = 1086313113130909791
# end of changelog



# on ready when bot comes online!
@bot.event
async def on_ready():
    # Set the bot's status to "listening to what the people want!"
    await bot.change_presence(activity=nextcord.Activity(type=nextcord.ActivityType.listening, name="what the people want!"))
    
    # Log in the console that the bot has successfully logged in
    print('Logged in as {0.user}'.format(bot))

# Get the changelog channel object
    changelog_channel = bot.get_channel(changelog_channel_id)


    
# Send the new changelog message to the channel
    embed = nextcord.Embed(title="Changelog for Wadder", description=changelog, color=0xFF5733)
    embed.add_field(name="Developer", value="Wade#1781", inline=False)
    await changelog_channel.send(embed=embed)

    account = api.client.account.get_account()
    print(
        f"Connected to Apollo as {account['attributes']['first_name']} {account['attributes']['last_name']} (Username: {account['attributes']['username']})"
    )

# for the logging function (will move to slash_command folder later)
@bot.slash_command(name="setuplogger", description="Sets up the logger (Admin only)")
@commands.has_permissions(administrator=True)
async def setuplogger(interaction: nextcord.Interaction):
    await setup_logger(interaction)

@setuplogger.error
async def setuplogger_error(interaction: nextcord.Interaction, error):
    if isinstance(error, commands.MissingPermissions):
        await interaction.send("You do not have the required permissions to use this command.", ephemeral=True)
# end of the logging function set up




# end of the apollo commands!




# Add this line to run the setup_logging_events coroutine
bot.loop.create_task(setup_logging_events(bot))
# the apollo has to stay down here or it will bug out!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@bot.slash_command(description="Returns all servers on the panel.")
@application_checks.has_role(int(os.getenv("AUTHORIZED_ROLE")))
async def servers4(interaction):
    my_servers = api.client.servers.list_servers()
    server_list = ""
    for servers in my_servers:
        for server in servers:
            server_list += f"[{server['attributes']['name']}]({os.getenv('PANEL_URL')}/server/{server['attributes']['identifier']}) (`{server['attributes']['identifier']}`)\n"
    embed = nextcord.Embed(
        title="Apollo Servers",
        description=f"Here are your servers hosted on Apollo Panel:\n\n{server_list or '**Unable to find any servers associated with this account.**'}",
        color=0xFFEA00,
    )
    await interaction.response.send_message(embed=embed)


@bot.slash_command(description="Send a console command to a server.")
@application_checks.has_role(int(os.getenv("AUTHORIZED_ROLE")))
async def command4(
    interaction,
    server_id: str = nextcord.SlashOption(description="Server ID"),
    command: str = nextcord.SlashOption(
        description="Command to send to the server's console"
    ),
):
    api.client.servers.send_console_command(server_id, command)

    embed = nextcord.Embed(
        title="Success",
        description=f":white_check_mark: Successfully sent command `{command}` to server `{server_id}`.",
        color=0x00FF04,
    )
    await interaction.response.send_message(embed=embed)


@bot.slash_command(description="Send a power action to a server.")
@application_checks.has_role(int(os.getenv("AUTHORIZED_ROLE")))
async def power4(
        interaction,
        server_id: str = nextcord.SlashOption(description="Server ID"),
        power_action: str = nextcord.SlashOption(
            description="Select a power action",
            choices={
                "Start": "start",
                "Restart": "restart",
                "Stop": "stop",
                "Kill": "kill",
            },
        ),
    ):
        api.client.servers.send_power_action(server_id, power_action)

        embed = nextcord.Embed(
            title="Success",
            description=f":white_check_mark: Successfully sent power action `{power_action}` to server `{server_id}`.",
            color=0x00FF04,
        )
        await interaction.response.send_message(embed=embed)
        




@bot.command()
async def load (ctx, extension):
    bot.load_extension(f"cogs.{extension}")
    await ctx.send("Loaded cog!")
    
@bot.command()
async def unload (ctx, extension):
    bot.unload_extension(f"cogs.{extension}")
    await ctx.send("Unloaded cog!")    


@bot.command()
async def reload (ctx, extension):
    bot.reload_extension(f"cogs.{extension}")
    await ctx.send("Reloaded cog!")

# Load cogs (i Know this is ugly but any other way it would not load anytihng else)
bot.load_extension("cogs.cog1")
bot.load_extension("cogs.cog2")
bot.load_extension("cogs.hangmancog")
bot.load_extension("cogs.cards")
bot.load_extension("cogs.math")
bot.load_extension("cogs.starboard")
bot.load_extension("cogs.gamenight")
bot.load_extension("cogs.customcm")
bot.load_extension("cogs.rr")
bot.load_extension("cogs.cursys")
bot.load_extension("cogs.levelsystem")
bot.load_extension("cogs.birthday")
bot.load_extension("cogs.antispam")
bot.load_extension("cogs.archive")
bot.load_extension("cogs.cry")
bot.load_extension("cogs.stockprice") # upated 
bot.load_extension("cogs.stockgame") # update
bot.load_extension("cogs.marknews")
bot.load_extension("cogs.userana") # this needs more work
bot.load_extension("cogs.tiksys")
bot.load_extension("cogs.poller")
bot.load_extension("cogs.auctionhouse")
bot.load_extension("cogs.cushelp")
bot.load_extension("cogs.remsys")
bot.load_extension("cogs.cloud")
bot.load_extension("cogs.welcomsg")
bot.load_extension("cogs.statconts")
bot.load_extension("cogs.bugreports")
bot.load_extension("cogs.eco")
bot.load_extension("cogs.quantcom")#1
bot.load_extension("cogs.mac")#2
bot.load_extension("cogs.voice")#3
bot.load_extension("cogs.vc")#4S
bot.load_extension("cogs.mcstatus")# - instead of refresh button make it on asyno done
bot.load_extension("cogs.arma") # - same as mcstatus done
bot.load_extension("cogs.spetotext")



bot.run(TOKEN)    
