import discord
import os
from dotenv import load_dotenv
import requests
import openai_model

load_dotenv()
intents = discord.Intents.default()
intents.message_content = True

resume = [None]
index = None
model = openai_model.Model()
loaded_resume = False

def run_discord_bot():
    print('loop')
    TOKEN = os.getenv('BOT_TOKEN')
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'{client.user} is now running')

        
    
    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        
        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)
        attatchments = message.attachments
        
        print(f"{username} said: '{user_message}' ({channel})")

        # loading resumes [!load]
        if user_message[:6] == '!load':
            user_message = user_message[5:]
            await load_resume(message, user_message, attatchments)

        # # general feedback [!general_feedback]
        # if user_message[:17] == '!general_feedback':
        #     user_message = user_message[17:]
        #     await general_response(message, user_message)

        # specific feedback !specific_feedback <section>/<type>
        # type = 'personal project', 'internship', 'award', etc..
        if user_message[:10] == '!specific ':
            user_message = user_message[10:]
            print(user_message)
            await specific_response(message, user_message)

        # # rewriting points [!rewrite "instructions"]
        # if user_message[:6] == '!rewrite ':
        #     instructions = user_message[6:]
        #     await rewrite(message, instructions)

    client.run(TOKEN)

# async def general_response(message, user_message):
#     await message.channel.send(await model.general_feedback())

async def specific_response(message, user_message):
    user_message = user_message.split('/')
    if (len(user_message) != 2):
        await message.channel.send('formatted request wrong, try !specific_feedback <section> <type>')
    else:
        await message.channel.send(await model.specific_feedback(user_message[0], user_message[1]))

async def load_resume(message, user_message, attatchments):
    print(attatchments)
    if (len(attatchments) > 1):
        await message.channel.send(f'Please send only one document at a time. sent: {len(attatchments)}')
    elif (len(attatchments) == 0):
        # check empty attachments
        await message.channel.send('Dont forget to attatch your resume!')
    else:
        document = attatchments[0]
        document_name = document.filename
        document_url = document.url
        document_id = document.id
        
        resume[0] = document_url

        # only accepts pdfs
        if (document.content_type != 'application/pdf'):
            await message.channel.send('I can only read pdfs...')
        else:
            print('downloading resume...')
            await download_pdf(document_url)
            loaded_resume = True
            await model.create_router_agent()

async def download_pdf(url):
    response = requests.get(url, allow_redirects=True)

    with open("resumes/resume.pdf",'wb') as f:
        f.write(response.content)            
    

            
            
# async def send_message(message, user_message, is_private):
#     try:
#         response = responses.handle_response(user_message)
#         await message.author.send(response) if is_private else await message.channel.send(response)
#     except Exception as e:
#         print(e)
    