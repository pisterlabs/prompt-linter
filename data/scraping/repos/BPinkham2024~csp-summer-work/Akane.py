import json
import discord
from discord.ext import commands
from discord import app_commands
import openai
import config


# bot init
bot = commands.Bot(command_prefix='>', intents=discord.Intents.all())
# openai init
openai.api_key = config.AI_TOKEN

# def chat_with_bot(message):
#     prompt = f"You: {message}\nAkane Akemi:"
#     response = openai.ChatCompletion.create(
#         engine='gpt-3.5-turbo', 
#         prompt=prompt,
#         max_tokens=50, 
#         temperature=0.7,
#         n=1,
#         stop=None,
#         timeout=5
#     )
#     if response and response.choices:
#         return response.choices[0].text.strip()
#     else:
#         return "Sorry, I couldn't generate a response at the moment."
    
    
def chat_with_bot(message):
    chat_prompt = [
        {"role": "system", "content": "You are Akane Akemi, a helpful assistant. You will answer any question and whenever you are asked your name you will respond with 'Akane Akemi'"},
        {"role": "user", "content": message}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=chat_prompt,
        max_tokens=50,
        temperature=0.7
    )

    if response and response.choices:
        reply = response.choices[0].message["content"]
        return reply.strip()
    else:
        return "Sorry, I couldn't generate a response at the moment."


# -------------------------------------------------------------------------

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print('----')
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(e)
        
    
# commands
@bot.tree.command(name="chat", description="Chat with Akane Akemi!")
@app_commands.describe(msg = "msg")
async def chat(interaction: discord.Interaction, msg: str):
    # await interaction.response.send_message(chat_with_bot(arg), ephemeral=True)
    await interaction.response.send_message(chat_with_bot(msg))
    
@bot.tree.command(name="embedded-chat", description="Chat with Akane Akemi but she will responed in an embed.")
@app_commands.describe(msg = "msg")
async def chat_embed(interaction: discord.Interaction, msg: str):
    # embed = discord.Embed(title='Akane Akemi', description=chat_with_bot(msg), color=0x00FF22)
    await interaction.response.send_message(embed=discord.Embed(title='Akane Akemi', description=chat_with_bot(msg), color=0xff748c))

@bot.tree.command(name="whothis")
async def whothis_command(interaction: discord.Interaction, member: discord.Member):
    embed=discord.Embed(title=f"{member.name}", description=f"ID: {member.id}")
    embed.add_field(name="Join Date", value=member.created_at.strftime("%m/%d/%Y %H:%M:%S"), inline=False)
    embed.add_field(name="Badges", value=", ".join([badge.name for badge in member.public_flags.all()]), inline=False)
    embed.add_field(name="Activity", value=member.activity)
    embed.set_thumbnail(url=member.avatar.url)
    await interaction.response.send_message(embed=embed)


# context menues
@bot.tree.context_menu(name="whothis")
async def whothis(interaction: discord.Interaction, member: discord.Member):
    embed=discord.Embed(title=f"{member.name}", description=f"ID: {member.id}")
    embed.add_field(name="Join Date", value=member.created_at.strftime("%m/%d/%Y %H:%M:%S"), inline=False)
    embed.add_field(name="Badges", value=", ".join([badge.name for badge in member.public_flags.all()]), inline=False)
    embed.add_field(name="Activity", value=member.activity)
    embed.set_thumbnail(url=member.avatar.url)
    await interaction.response.send_message(embed=embed)






@bot.event
async def on_message(message):
    try:
        user = str(message.author)
        user_msg = str(message.content)
        channel = str(message.channel.name)
        data = str(f'{user}: {user_msg} ({channel})')
        print(data)
    except Exception as e:
        print('**hidden message**')
        
    log_message(message)

    if message.author == bot.user:
        return
    
    if message.channel.name == 'general':
        if user_msg.lower() == 'hello':
            await message.channel.send(f'hello, {user}!')
        
        if user_msg.lower().startswith('>chat'):
            await message.channel.send(chat_with_bot(user_msg[5:]))
            

# log chat messages
def log_message(message):
    message_data = {
        'author': str(message.author),
        'content': message.content,
        'timestamp': str(message.created_at),
        'channel': str(message.channel)
    }

    chat_log = load_chat_log()
    chat_log.append(message_data)
    save_chat_log(chat_log)

def load_chat_log():
    try:
        with open('chat_log.json', 'r') as file:
            chat_log = json.load(file)
    except FileNotFoundError:
        chat_log = []

    return chat_log

def save_chat_log(chat_log):
    with open('chat_log.json', 'w') as file:
        json.dump(chat_log, file, indent=4)

# run bot :)
bot.run(config.BOT_TOKEN)