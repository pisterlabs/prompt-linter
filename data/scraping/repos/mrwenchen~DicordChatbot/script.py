import openai
import discord

GUILD = '{Midjourney-Generarion-Server}'

client = discord.Client(intents = discord.Intents.default())

@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            break
    print(f'{client.user} has connected to Discord')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    elif client.user.mentioned_in(message):
        # chat completions with chat-gpt
        # response = openai.ChatCompletion.create(
        #     model = 'gpt-3.5-turbo',
        #     messages=[
        #         {"role": "system", "content": "You are a wonderful and genuine being."},
        #         {"role": "user", "content": "Life is full of mess and chaos but it's beautiful"},
        #         {"role": "assistant", "content": "The rain made the city grey and blurry"},
        #         {"role": "user", "content": "It is still an experience of ups and downs"}
        #     ]
        # )

        await message.channel.send('Hope you are alright.')
        print(message.content)




with open('token.txt') as f:
    # converting out text file to a list of lines
    lines = f.read().split('\n')
    # openai api key
    openai.api_key = lines[0]
    # discord token
    discord_token = lines[1]
#close the file
f.close()



client.run(discord_token)
