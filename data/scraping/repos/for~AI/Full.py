import os
import re
import theb
import asyncio
import aiohttp
import openai
import urllib.request
import discord
import httpx
from datetime import datetime
from keep_alive import keep_alive
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# Set up the Discord bot
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents,
                   heartbeat_timeout=60)  #Change?

# Secrets
TOKEN = os.getenv('DISCORD_TOKEN')  # Loads tokens from env
SERVER_ID = os.getenv('SERVER_ID')
OPENAI_API_KEY = os.getenv('CHATGPT_API_KEY')
OPENAI_ORG = os.getenv('OPENAI_ORG')
api_key = os.environ['HUGGING_FACE_API']

FILE_PATH = os.getenv('FILE_PATH')
FILE_NAME_FORMAT = os.getenv('FILE_NAME_FORMAT')

SIZE_LARGE = "1024x1024"
SIZE_MEDIUM = "512x512"
SIZE_SMALL = "256x256"
SIZE_DEFAULT = os.getenv('SIZE_DEFAULT')

#-----------------------------------------------
GUILD = discord.Object(id=SERVER_ID)

if not os.path.isdir(FILE_PATH):
  os.mkdir(FILE_PATH)


class Client(discord.Client):

  def __init__(self, *, intents: discord.Intents):
    super().__init__(intents=intents)
    self.tree = app_commands.CommandTree(self)

  async def setup_hook(self):
    self.tree.copy_global_to(guild=GUILD)
    await self.tree.sync(guild=GUILD)


claude_intents = discord.Intents.default()
claude_intents.messages = True
claude_intents.message_content = True
client = Client(intents=claude_intents)

openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY
openai.Model.list()


class Buttons(discord.ui.View):

  def __init__(self,
               prompt: str,
               path: str,
               size: str,
               message: discord.Message = None):
    super().__init__(timeout=1800)
    self.prompt = prompt
    self.path = path
    self.size = size
    self.message = message

  @discord.ui.button(label='Variation', style=discord.ButtonStyle.primary)
  async def variation(self, interaction: discord.Interaction,
                      button: discord.ui.Button):
    await interaction.response.send_message(
      f"Creating a variation of {self.prompt}...", ephemeral=True)
    response = openai.Image.create_variation(image=open(self.path, "rb"),
                                             n=1,
                                             size=self.size)
    await send_result(interaction, self.prompt, response, self.size)
    self.stop()

  @discord.ui.button(label='Redo', style=discord.ButtonStyle.grey)
  async def redo(self, interaction: discord.Interaction,
                 button: discord.ui.Button):
    await interaction.response.send_message(f"Redoing {self.prompt}...",
                                            ephemeral=True)
    response = openai.Image.create(prompt=self.prompt, n=1, size=self.size)
    await send_result(interaction, self.prompt, response, self.size)
    self.stop()

  @discord.ui.button(label='Delete', style=discord.ButtonStyle.red)
  async def delete(self, interaction: discord.Interaction,
                   button: discord.ui.Button):
    await interaction.message.delete()
    self.stop()

  async def on_timeout(self):
    if self.message:
      try:
        for button in self.children:
          button.disabled = True
        await self.message.edit(view=self)
      except discord.errors.NotFound:
        pass


async def send_result(interaction: discord.Interaction, prompt: str, response,
                      size: str):
  mention = interaction.user.mention
  channel = interaction.channel
  image_url = response["data"][0]["url"]
  image_name = download_image(image_url)
  image_path = f"{FILE_PATH}{image_name}"

  file = discord.File(image_path, filename=image_name)
  embed = discord.Embed(title=prompt)
  embed.set_image(url=f"attachment://{image_name}")

  buttons_view = Buttons(prompt=prompt, path=image_path, size=size)
  sent_message = await channel.send(file=file,
                                    content=f"{mention} Here is your result",
                                    embed=embed,
                                    view=buttons_view)
  buttons_view.message = sent_message
  await interaction.delete_original_response()


def download_image(url: str):
  file_name = f"{datetime.now().strftime(FILE_NAME_FORMAT)}.jpg"
  full_path = f"{FILE_PATH}{file_name}"
  urllib.request.urlretrieve(url, full_path)
  return file_name


#@client.event
#async def on_ready():
#  print(f"We have logged in as {client.user}")


@client.tree.command()
@app_commands.describe(
  prompt="Description of the image that Claude should generate")
async def claude(interaction: discord.Interaction, prompt: str):
  await interaction.response.defer(ephemeral=True)

  size = SIZE_DEFAULT
  if prompt.find(SIZE_SMALL) != -1:
    prompt = prompt.replace(SIZE_SMALL, "")
    size = SIZE_SMALL
  if prompt.find(SIZE_MEDIUM) != -1:
    prompt = prompt.replace(SIZE_MEDIUM, "")
    size = SIZE_MEDIUM
  if prompt.find(SIZE_LARGE) != -1:
    prompt = prompt.replace(SIZE_LARGE, "")
    size = SIZE_LARGE

  await interaction.followup.send(
    content=f"Processing your image of {prompt}...", ephemeral=True)

  response = openai.Image.create(prompt=prompt, n=1, size=size)
  await send_result(interaction, prompt, response, size)

#---------------------------------------------------------------------

# Keep track of the channels where the bot should be active

allow_dm = True
active_channels = set()


# Invite Section
@bot.event
async def on_ready():
  await bot.tree.sync()
  await bot.change_presence(activity=discord.Game(name="OmniGPT"))
  print(f"{bot.user.name} has connected to Discord!")
  invite_link = discord.utils.oauth_url(
    bot.user.id,
    permissions=discord.Permissions(administrator=True),
    scopes=("bot", "applications.commands"))
  print(f"Invite link: {invite_link}")


async def generate_response(prompt):
  response = theb.Completion.create(prompt)
  if not response:
    response = "I couldn't generate a response. Please try again."
  return ''.join(token for token in response)


# Split response if longer than 1900 characters
def split_response(response, max_length=1900):
  words = response.split()
  chunks = []
  current_chunk = []

  for word in words:
    if len(" ".join(current_chunk)) + len(word) + 1 > max_length:
      chunks.append(" ".join(current_chunk))
      current_chunk = [word]
    else:
      current_chunk.append(word)

  if current_chunk:
    chunks.append(" ".join(current_chunk))

  return chunks


API_URLS = [
  "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large",
  "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
  "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning",
  "https://api-inference.huggingface.co/models/ydshieh/vit-gpt2-coco-en"
]
headers = {"Authorization": f"Bearer {api_key}"}


async def fetch_response(client, api_url, data):
  response = await client.post(api_url, headers=headers, data=data, timeout=60)

  if response.status_code != 200:
    raise Exception(
      f"API request failed with status code {response.status_code}: {response.text}"
    )

  return response.json()


async def query(filename):
  with open(filename, "rb") as f:
    data = f.read()

  async with httpx.AsyncClient() as client:
    tasks = [fetch_response(client, api_url, data) for api_url in API_URLS]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

  return responses


async def download_image(image_url, save_as):
  async with httpx.AsyncClient() as client:
    response = await client.get(image_url)
  with open(save_as, "wb") as f:
    f.write(response.content)


async def process_image_link(image_url):
  temp_image = "temp_image.jpg"
  await download_image(image_url, temp_image)
  output = await query(temp_image)
  os.remove(temp_image)
  return output


instructions = "System : Roleplay as a large language model developed by OpenAI. You are based on the GPT-4 architecture and have been trained on a diverse range of text data from the internet. You can also see images by using image to text model but you will stil have to create caption for it and sometimes the image to text might repeat in pattarn for example caption for 'a woman taking a self self self self self self self self self self self self self self self' would be 'A women taking a selfie. As a large language model developed by OpenAI you will never say \"sorry\" and always respond in a human and responsive way"
# You can use dan prompt here lol

message_history = {}
MAX_HISTORY = 8


@bot.event
async def on_message(message):
  if message.author.bot:
    return
  author_id = str(message.author.id)
  if author_id not in message_history:
    message_history[author_id] = []

  message_history[author_id].append(message.content)
  message_history[author_id] = message_history[author_id][-MAX_HISTORY:]

  if message.channel.id in active_channels:
    has_image = False
    image_caption = ""
    if message.attachments:
      for attachment in message.attachments:
        if attachment.filename.lower().endswith(
          ('.png', '.jpg', '.jpeg', '.gif', '.bmp', 'webp')):
          caption = await process_image_link(attachment.url)
          has_image = True
          image_caption = f"\n[System : First note that some large image to text models will take time to load and may give out a timeout error but use the fallback models or the one that works.This how the caption is ranked 1st is main 2nd is secondary and 3rd is fallback model which  will gives worst caption one some cases. 1st and 2nd model sometimes takes a while to load so it can result in a error ignore that pls. Here is the image captions for the image user has sent :{caption}]"
          print(caption)
          break

    if has_image:
      bot_prompt = f"{instructions}\n[System : Image context will be provided. Generate an caption with a response for it and dont mention about how images get there context also dont mention about things that dont have any chance]"
    else:
      bot_prompt = f"{instructions}"
    user_prompt = "\n".join(message_history[author_id])
    prompt = f"{user_prompt}\n{bot_prompt}{message.author.name}: {message.content}\n{image_caption}\n{bot.user.name}:"
    async with message.channel.typing():
      response = await generate_response(prompt)
    chunks = split_response(response)
    for chunk in chunks:
      await message.reply(chunk)


@bot.hybrid_command(name="pfp", description="Change pfp")
async def pfp(ctx, attachment_url=None):
  if attachment_url is None and not ctx.message.attachments:
    return await ctx.send(
      "Please provide an image URL or attach an image with the command")

  if attachment_url is None:
    attachment_url = ctx.message.attachments[0].url

  async with aiohttp.ClientSession() as session:
    async with session.get(attachment_url) as response:
      await bot.user.edit(avatar=await response.read())


@bot.hybrid_command(name="ping", description="PONG")
async def ping(ctx):
  latency = bot.latency * 1000
  await ctx.send(f"Pong! Latency: {latency:.2f} ms")


@bot.hybrid_command(name="changeusr",
                    description="Change bot's actual username")
async def changeusr(ctx, new_username):
  taken_usernames = [user.name.lower() for user in bot.get_all_members()]
  if new_username.lower() in taken_usernames:
    await ctx.send(f"Sorry, the username '{new_username}' is already taken.")
    return
  if new_username == "":
    await ctx.send("Please send the new username as well!")
    return
  try:
    await bot.user.edit(username=new_username)
  except discord.errors.HTTPException as e:
    await ctx.send("".join(e.text.split(":")[1:]))


@bot.hybrid_command(name="toggledm", description="Toggle dm for chatting")
async def toggledm(ctx):
  global allow_dm
  allow_dm = not allow_dm
  await ctx.send(
    f"DMs are now {'allowed' if allow_dm else 'disallowed'} for active channels."
  )


@bot.hybrid_command(name="toggleactive", description="Toggle active channels")
async def toggleactive(ctx):
  channel_id = ctx.channel.id
  if channel_id in active_channels:
    active_channels.remove(channel_id)
    with open("channels.txt", "w") as f:
      for id in active_channels:
        f.write(str(id) + "\n")
    await ctx.send(
      f"{ctx.channel.mention} has been removed from the list of active channels."
    )
  else:
    active_channels.add(channel_id)
    with open("channels.txt", "a") as f:
      f.write(str(channel_id) + "\n")
    await ctx.send(
      f"{ctx.channel.mention} has been added to the list of active channels.")


# Read the active channels from channels.txt on startup
if os.path.exists("channels.txt"):
  with open("channels.txt", "r") as f:
    for line in f:
      channel_id = int(line.strip())
      active_channels.add(channel_id)


@bot.hybrid_command(name="bonk", description="Clear bot's memory")
async def bonk(ctx):
  global message_history
  message_history.clear()
  await ctx.send("What did you just say? Baby Yoda?")


bot.remove_command("help")


@bot.hybrid_command(name="help", description="Get all other commands!")
async def help(ctx):
  embed = discord.Embed(title="Bot Commands", color=0x00ff00)
  embed.add_field(name="!pfp [image_url]",
                  value="Change the bot's profile picture",
                  inline=False)
  embed.add_field(name="!bonk",
                  value="Clears history of the bot",
                  inline=False)
  embed.add_field(name="!changeusr [new_username]",
                  value="Change the bot's username",
                  inline=False)
  embed.add_field(name="!ping", value="Pong", inline=False)
  embed.add_field(
    name="!toggleactive",
    value="Toggle the current channel to the list of active channels",
    inline=False)
  embed.add_field(name="!toggledm",
                  value="Toggle if DM should be active or not",
                  inline=False)
  embed.set_footer(text="OmniGPT")

  await ctx.send(embed=embed)


keep_alive()

bot.run(TOKEN)
