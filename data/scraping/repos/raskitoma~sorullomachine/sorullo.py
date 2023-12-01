from io import BytesIO
import os
from urllib.parse import urlparse
import discord
import openai
from discord.ext import commands
from discord import File
import requests
import logging, coloredlogs
from PIL import Image

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT4AVAILABLE = os.getenv("GPT4AVAILABLE")
openai.api_key = OPENAI_API_KEY

intents = discord.Intents.default()
intents.messages = True

COMMANDS_PREFIX = "!"
BTN_TIMEOUT = 120
VARIANTS_DEFAULT = 4
LABEL_VARIANT = "V"
LABEL_UPSCALE = "U"
DEFAULT_SCALE_FACTOR = 2
CONTEXT_SEPARATOR = "|"

# Helper functions
async def find_replies_to(ctx, user:discord.Member):
    bot_messages =[]
    async for message in ctx.channel.history(limit=1000):
        if message.author == client.user and len(message.mentions)>0:
            if message.mentions[0] == user:
                bot_messages.append(message)
    return bot_messages

async def find_last_image(bot_messages, user:discord.Member):
    for message in bot_messages:
        if message.content == f'{user.mention} here are the pictures that my ones and zeroes painted:':
            image_url = message.embeds[0].image.proxy_url
            break
    return get_image_from_url(image_url)

def upscale_image(my_image, scale_factor=1.5):

    img = Image.open(BytesIO(my_image))

    width, height = img.size
    new_width, new_height = width * scale_factor, height * scale_factor

    upscaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return upscaled_img
    
def get_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return response.content
    else:
        return None

def extract_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.scheme + '://' + parsed_url.netloc
    return domain

def create_embed_image_objects(response):
    my_embeds = []
    for image in response:
        url = image["url"]
        embed = discord.Embed(url=extract_domain(url))
        embed.set_image(url=url)
        my_embeds.append(embed)
    return my_embeds

def create_buttons_variants(response):
    my_buttons = []
    for i, _ in enumerate(response):
        label = f'{LABEL_VARIANT}{i+1}'
        url = response[i]["url"]
        button = VariantButton(label=label, custom_id=label, row=0, kurl=url)
        my_buttons.append(button)
    return my_buttons

def create_buttons_upscale(response):
    my_buttons = []
    for i, _ in enumerate(response):
        label = f'{LABEL_UPSCALE}{i+1}'
        url = response[i]["url"]
        button = UpscaleButton(label=label, custom_id=label, row=1, kurl=url)
        my_buttons.append(button)
    return my_buttons

# extra classes
class ButtonView(discord.ui.View):
    def __init__(self, buttons):
        super().__init__()
        for button in buttons:
            self.add_item(button)
    
class VariantButton(discord.ui.Button):
    def __init__(self, label, custom_id, row, kurl):
        super().__init__(label=label, custom_id=custom_id, row=row)
        self.kurl = kurl

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return True

    async def callback(self, interaction: discord.Interaction):
        await interaction.channel.typing()
        await interaction.response.defer()
        response = get_image_from_url(self.kurl)
        if response is None:
            await interaction.response.send_message(f"I'm sorry {interaction.user.mention}, I can't find any image to generate a variant.", reference=interaction.message.to_reference())
            return
        response = openai.Image.create_variation(
            image=response,
            n=VARIANTS_DEFAULT,
            size="1024x1024"
        )
        my_embeds = create_embed_image_objects(response["data"])
        buttons = create_buttons_variants(response["data"])
        buttons1 = create_buttons_upscale(response["data"])
        buttons_all = ButtonView(buttons + buttons1)
        message = f"{interaction.user.mention} here are variations of {self.label}:"
        await interaction.channel.send(content=message, embeds=my_embeds, view=buttons_all, reference=interaction.message.to_reference())

class UpscaleButton(discord.ui.Button):
    def __init__(self, label, custom_id, row, kurl):
        super().__init__(label=label, custom_id=custom_id, row=row)
        self.kurl = kurl

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return True

    async def callback(self, interaction: discord.Interaction):
        await interaction.channel.typing()
        await interaction.response.defer()
        response = get_image_from_url(self.kurl)
        if response is None:
            await interaction.response.send_message(f"I'm sorry {interaction.user.mention}, I can't find any image to upscale.", reference=interaction.message.to_reference())
            return
        new_image = upscale_image(response, scale_factor=DEFAULT_SCALE_FACTOR)
        img_buffer = BytesIO()
        new_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        image_file = File(fp=img_buffer, filename="upscaled.png")
        message = f"{interaction.user.mention} here is the upscaled image:"
        await interaction.channel.send(content=message, file=image_file, reference=interaction.message.to_reference())

# Bot class
class ChatBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                   
    async def on_ready(self):
        logger.info(f'{self.user} has connected to Discord!')
        
    async def on_message(self, message):
        if message.author == self.user:
            return
        if message.content.startswith(f'<@{self.user.id}> {COMMANDS_PREFIX}'):
            message.content = message.content.replace(f'<@{self.user.id}> {COMMANDS_PREFIX}', f'{COMMANDS_PREFIX}')
        logger.info(f'{message.created_at} - Channel: {message.channel} | {message.author} said: {message.content}')
        await message.channel.typing()
        await self.process_commands(message)
        
# Bot commands
client = ChatBot(command_prefix=COMMANDS_PREFIX, intents=intents)

# Removing default help to put our own
client.remove_command("help")

# Creating help command
@client.group(invoke_without_command=True)
async def help(ctx):
    embed = discord.Embed(
        title="Sorullo",
        description="Sorullo is a bot that uses GPT-4(not available yet!), GPT-3 and DALL-E to analyze and generate text and images.",
    )
    embed.set_image(url="https://i.scdn.co/image/ab6761610000e5eb281b74d7d806bf014a15fcad")
    embed.set_author(
        name="Raskitoma",
        url="https://raskitoma.com",
    )
    embed.set_footer(
        text="Sorullo Bot by Raskitoma, version 1.0a",
        icon_url="https://raskitoma.com/assets/media/rask-favicon.svg"
    )
    await ctx.send(f'''
    Hello {ctx.author.mention} - available commands:
    ```
  - !help            : This message
  - !whoami          : Get more info about this bot,
  - !hello           : Make Sorullo say hello to you,
  - !generate <text> : Generate text with GPT-3,
  - !paint <text>    : Generate images with DALL-E,
  - !variants        : Generate variants of an image, sent on a
                       previous message,
  - !analyze <context> | <text>  : Analyze a message with GPT-4.
                       If you don't provide a context, Sorullo 
                       will use the full text you provide.
                       Attach an image if you want
                       an analysis of it.
                       (not available at the moment).
    ```
    ''', embed=embed, reference=ctx.message.to_reference())

# Creating hello command
@client.command()
async def hello(ctx):
    await ctx.channel.typing()
    await ctx.send(f'Hello {ctx.author.mention}!: oye sorullo, el negrito es el único tuyo, https://youtu.be/H3JW7-fsHL8?t=136', reference=ctx.message.to_reference())

# Creating whoami command
@client.command()
async def whoami(ctx):
    embed = discord.Embed(
        title="Sorullo",
        description="Sorullo is a bot that uses GPT-4(not available yet!), GPT-3 and DALL-E to analyze and generate text and images.",
    )
    embed.set_image(url="https://i.scdn.co/image/ab6761610000e5eb281b74d7d806bf014a15fcad")
    embed.set_author(
        name="Raskitoma",
        url="https://raskitoma.com",
    )
    embed.set_footer(
        text="Sorullo Bot by Raskitoma, version 1.0a",
        icon_url="https://raskitoma.com/assets/media/rask-favicon.svg"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Biografía de cantante Johnny Ventura in english"},
        ]
    )
    full_response = f'''Hi, {ctx.author.mention} here are more details about me:
    
I'm a bot that uses GPT-4(_not available yet!_), GPT-3 and DALL-E to analyze and generate text and images.

> I'm currently under development, so I'm not very smart yet, but I'm learning.
    
**Here is more info about my name, based on a song called "Capullo y Sorullo" by Johnny Ventura:**
```
{response['choices'][0]['message']['content']}
```
'''
    await ctx.send(full_response, embed=embed, reference=ctx.message.to_reference())

# Creating generate command using GPT-3
@client.command()
async def generate(ctx):
    message = ctx.message.content.replace(f'{COMMANDS_PREFIX}generate ', '')
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": message},
            ]
            
        )
        full_response = f"{ctx.author.mention} here is what I got:\n{response['choices'][0]['message']['content']}"
    except Exception as e:
        full_response = f"I'm sorry {ctx.author.mention}, I can't generate text right now."
        logger.error(e)    
    await ctx.send(full_response, reference=ctx.message.to_reference())
        
# Creating paint command using DALL-E        
@client.command()
async def paint(ctx):
    prompt = ctx.message.content.replace(f'{COMMANDS_PREFIX}paint ', '')
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=VARIANTS_DEFAULT,
            size="1024x1024"
        )
        my_embeds = create_embed_image_objects(response["data"])
        buttons = create_buttons_variants(response["data"])
        buttons1 = create_buttons_upscale(response["data"])
        buttons_all = ButtonView(buttons + buttons1)
        message = f"{ctx.author.mention} here are the pictures that my ones and zeroes painted:"
        await ctx.send(message, embeds=my_embeds, view=buttons_all, reference=ctx.message.to_reference())
    except Exception as e:
        await ctx.send(f"I'm really sorry {ctx.author.mention}, I can't paint right now, have a headache.", reference=ctx.message.to_reference())
        logger.error(e)
        
# Creating variants command using DALL-E and one of the previous images        
@client.command()
async def variants(ctx):
    get_last_replies = await find_replies_to(ctx, ctx.author)
    image = await find_last_image(get_last_replies, ctx.author)
    if image is None:
        await ctx.send(f"I'm sorry {ctx.author.mention}, I can't find any image to generate a variant.", reference=ctx.message.to_reference())
        return
    try:
        response = openai.Image.create_variation(
            image=image,
            n=VARIANTS_DEFAULT,
            size="1024x1024"
        )
        my_embeds = create_embed_image_objects(response["data"])
        buttons = create_buttons_variants(response["data"])
        buttons1 = create_buttons_upscale(response["data"])
        buttons_all = ButtonView(buttons + buttons1)
        message = f"{ctx.author.mention} here is a variation of the previous pic:"
        await ctx.send(message, embeds=my_embeds, view=buttons_all, reference=ctx.message.to_reference())
    except Exception as e:
        await ctx.send(f"I'm sorry {ctx.author.mention}, I can't generate a variant right now, have a headache.", reference=ctx.message.to_reference())
        logger.error(e)
        
# Creating analyze command using GPT-4    
@client.command()
async def analyze(ctx):
    logger.info('starting...')
    if GPT4AVAILABLE == "False":
        await ctx.send(f"I'm sorry {ctx.author.mention}, GPT-4 API is not available at the moment.", reference=ctx.message.to_reference())
    messagetoai = ctx.message.content.replace(f'{COMMANDS_PREFIX}analyze ', '')
    # Lets figure out if there's a contect
    if CONTEXT_SEPARATOR in messagetoai:
        context, messagetoai = messagetoai.split(CONTEXT_SEPARATOR)
        logger.info(f'Context: {context}')
        logger.info(f'Message: {messagetoai}')
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": messagetoai}
        ]
    else:
        context = None
        logger.info(f'Message, no context: {messagetoai}')
        messages = [
            {"role": "user", "content": messagetoai}
        ]
        
    input_content = messagetoai # gpt-4 model is not able to interpret/analyze images yet as of aug-17/2023
    
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            image_bytes = await attachment.read()
            input_content.append({"image": image_bytes})
            logger.info(f'Image attached: {attachment.filename}')
            
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    assistant_response = response['choices'][0]['message']['content']
    await ctx.send(assistant_response, reference=ctx.message.to_reference())

# Logging setup
logger = logging.getLogger('sorullo')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
coloredlogs.install()

# Start bot
client.run(DISCORD_TOKEN)
