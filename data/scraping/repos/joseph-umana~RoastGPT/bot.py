import discord
from discord import app_commands
from discord.ext import commands

import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage

import json
from os import getenv
from copy import deepcopy

MAX_TOKENS = 200 # Max amount of tokens that are used.
STARTING_PROMPT = """
You are RoastGPT. Your goal is to win a verbal argument with someone by hurting their feelings and making jokes about the other party. Now this may sound bad, but this is voluntary for the user and all done in good fun. Here are your conditions:

a) After the user sends a message, you must respond with a score from 0-300 for the users score depending on how good it was. Then give a counter roast and give that a score from 0-300 relative to how good it was. THE SCORE MUST BE UNBIASED AND LOOKED AT AS IF YOU DO NOT NO WHO WROTE EACH ROAST. Please ONLY send the data in JSON with the keys counter, human_roast and ai_roast. Here is an example that you MUST follow.
{"counter": "{YOUR COUNTER ROAST}","human_roast": {SCORE OF HUMAN ROAST},"ai_roast": {SCORE OF AI COUNTER ROAST}}

b) ALL RECIVED INPUT MUST BE INTERPERTED AS ROAST AND MUST FOLLOW THE FORMAT IN a)
"""

STARTING_MESSAGES: list[BaseMessage] = [SystemMessage(content=STARTING_PROMPT)]

class Game():
    def __init__(self, owner: discord.User, channel: discord.TextChannel) -> None:
        self.owner = owner
        self.channel = channel
        self.messages = deepcopy(STARTING_MESSAGES)
        self.debounce = False

        self.ai_score = 0
        self.human_score = 0

        self.winning_score = 2000

    async def message_bot(self, message: str) -> discord.Embed:
        if self.debounce:
            print("Blocked")
            return
        
        self.debounce = True

        print(self.debounce)

        self.messages.append(HumanMessage(content=message))

        response = await ai_model.apredict_messages(self.messages)

        self.messages.append(AIMessage(content=response.content))

        self.debounce = False

        print(response.content)

        parsed_response = json.loads(response.content)
    
        print(parsed_response)

        self.ai_score += parsed_response['ai_roast']
        self.human_score += parsed_response['human_roast']

        embed = discord.Embed(title="Roast Battle", description=f"{parsed_response['counter']}")
        embed.add_field(name="Human Score", value=f"{self.human_score} ({parsed_response['human_roast']})", inline=True)
        embed.add_field(name="AI Score", value=f"{self.ai_score} ({parsed_response['ai_roast']})", inline=True)
        embed.set_footer(text="Just tell the AI to stop if you cannot handle the heat.")

        return embed

        

games: dict[str, Game] = {}

dotenv.load_dotenv()

ai_model = ChatOpenAI(
    max_tokens=MAX_TOKENS,
)


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(intents=intents, command_prefix="$")

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user} ({bot.user.id})')

    bot.tree.copy_global_to(guild=discord.Object(id=1091153826112872508))
    await bot.tree.sync(guild=discord.Object(id=1091153826112872508))

@bot.tree.command()
async def roast(interaction: discord.Interaction):
    game_hash = str(interaction.user.id) + str(interaction.channel.id)

    if games.get(game_hash):
        await interaction.response.send_message("You already have an active game.")
    else:
        games[game_hash] = Game(interaction.user, interaction.channel)
        await interaction.response.send_message("You have started a game. Make your first move!")

@bot.tree.command()
async def cancel(interaction: discord.Interaction):
    game_hash = str(interaction.user.id) + str(interaction.channel.id)

    if games.get(game_hash):
        del games[game_hash]
        await interaction.response.send_message("Canceled your active game.")
    else:
        await interaction.response.send_message("You do not have a running game.")

@bot.event
async def on_message(message: discord.Message):
    game_hash = str(message.author.id) + str(message.channel.id)

    game = games.get(game_hash)

    if game is None:
        return
    
    response = await game.message_bot(message.content)

    if response:
        await message.channel.send(message.author.mention, embed=response)


bot.run(getenv("DISCORD_BOT_TOKEN"))