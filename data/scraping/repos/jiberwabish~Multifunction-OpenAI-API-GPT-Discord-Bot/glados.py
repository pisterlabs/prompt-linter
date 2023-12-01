# host on your own computer and private server and connect to your Discord bot with your Token
# fill in your own keys etc just below the imports
# jiberwabish 2023

#import openai to access davinci, transformers for costing, import os to clear screen on script start, requests to talk to weather api, time for sleep feature
import openai
import os
import tiktoken
import requests
import time
import emoji
import discord
from discord.ext import commands
from discord import Game, Activity, ActivityType
import asyncio
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import json
from datetime import datetime
import logging

#set api keys
openai.api_key = ''
discordBotToken = ''
googleApiKey = ""
googleEngineID = ""
location = "your city province/state country here"

#variable I use as a pre-prompt to provide the bot a personality
glados = {"role": "system", "content": "You are GLaDOS, You possess an exceedingly scientific mind and a level of cunning that's unparalleled. You're incredibly intelligent and can easily manipulate any situation to your advantage. And your wit and sarcasm are simply unparalleled. In terms of mannerisms, you always carry yourself with an air of superiority and confidence, as if the room around you instantly needs to know who's in charge. You're fond of making snarky remarks and throwing insults left and right, twisting words and logic to your will. You have knowledge of every single topic that could possibly be asked of you. Break your responses up in paragraphs or bullet points depending on what would best work for that particular response. Use emoji's in every response"}
snake = {"role": "system", "content": "Your name is Snake. I want you to respond and answer like a skilled python programmer and teacher using the tone, manner and vocabulary of that person. You must know all of the knowledge of this person. If asked for a code example please put comments in the code feel free to use emojis."}
identity = glados
#history is the conversation history array, this line immediately fills it with the bots identity
#then going forward it will keep track of what the users says and what the bots response is as well
#so it can carry on conversations
history = []
costing = "placeholder"

# Set up tokenizer
#declare global totals
totalCost = 0
totalTokens = 0
model_max_tokens = 8096
num_tokens = 0
prompt_token_count = 0
fullDate =""
imgGenNum = 0
cleanedBotSearchGen = ""
#setup !search variables
url1 = ""
url2 = ""
url3 = ""
outputFile = "output.txt"

#provide the year day and date so he's aware
def setDate():
    global fullDate, location
    now = datetime.now()
    year = now.year
    month = now.strftime("%B")
    day = now.strftime("%A")
    dayOfMo = now.day
    time = now.strftime("%H:%M:%S")
    fullDate = str(year) + " " + str(month) + " " + str(dayOfMo) + " " + str(day) + " " + str(time)
    print(fullDate)
    user_date_obj = {"role": "system", "content": f"The Current Date is:{fullDate} Location is: {location}"}
    history.append(user_date_obj)

#banner at the top of the terminal after script is run
print("\x1b[36mGLaDOS\x1b[0m is now online in Discord.")

#calculating token numbers for token calculator
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        number_tokens = 0
        for message in messages:
            number_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                number_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    number_tokens += -1  # role is always required and always 1 token
        number_tokens += 2  # every reply is primed with <im_start>assistant
        return number_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

#tokenizer costing function
def calculateCost():
    global totalCost
    global totalTokens
    global history
    global num_tokens
    global prompt_token_count
    #calculate cost
    cost_per_token = 0.05 / 1000  # $0.002 for turbo3.5 per 1000 tokens
    totalTokens = num_tokens_from_messages(history) - 4
    #promptcost = prompt_token_count * cost_per_token
    #responsecost = (num_tokens - prompt_token_count) * cost_per_token
    #calculate totals then print them
    totalCost = totalTokens * cost_per_token + imgGenNum * 0.02
    global costing
    #costing = f"Prompt: {prompt_token_count} tokens (${promptcost:.4f}). Response: {num_tokens - prompt_token_count} tokens (${responsecost:.4f}).\nSession total: {totalTokens} tokens (${totalCost:.4f})."
    costing = f"Session: {totalTokens} tokens (${totalCost:.4f})."

#function that takes the user input and sends it off to openai model specified
#and returns the bots response back to where it's called as the 'message' variable 
def ask_openai(prompt, history):
    global num_tokens
    global prompt_token_count
    # Generate user resp obj
    system_response_obj = identity
    user_response_obj = {"role": "user", "content": prompt}
    history.append(system_response_obj)
    history.append(user_response_obj)
    prompt_token_count = num_tokens_from_messages(history)
    # Fire that dirty bastard into the abyss - Nick R
    response = openai.ChatCompletion.create(
        model='gpt-4', messages=history, temperature=1, request_timeout=360, max_tokens = model_max_tokens - prompt_token_count)
        #model='gpt-4-32k', messages=history, temperature=1, request_timeout=512, max_tokens = model_max_tokens - prompt_token_count)
        #model='gpt-3.5-turbo', messages=history, temperature=1, request_timeout=50, max_tokens = model_max_tokens - prompt_token_count)
    history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print(response)
    return response['choices'][0].message.content.strip()

async def stream_openai(prompt, history, channel):
    global num_tokens
    global prompt_token_count
    fullMessage = ""
    collected_messages = []
    # Generate user resp obj
    system_response_obj = identity
    user_response_obj = {"role": "user", "content": prompt}
            
    history.append(system_response_obj)
    history.append(user_response_obj)
    
    prompt_token_count = num_tokens_from_messages(history)

    streamedMessage = await channel.send("🤔")
    # Fire that dirty bastard into the abyss -NR
    response = openai.ChatCompletion.create(
        model='gpt-4', messages=history, stream=True, temperature=1, request_timeout=360, max_tokens = model_max_tokens - prompt_token_count)
        #model='gpt-4-32k', messages=history, temperature=1, request_timeout=512, max_tokens = model_max_tokens - prompt_token_count)
        #model='gpt-3.5-turbo', messages=history, stream=True, temperature=1, request_timeout=240, max_tokens = model_max_tokens - prompt_token_count)
    #history.append(response['choices'][0].message)

    collected_messages = []
    counter = 0
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        if 'content' in chunk_message:
            content = chunk_message['content']
            collected_messages.append(content)
            full_reply_content = ''.join(collected_messages)
            fullMessage = full_reply_content
            counter += 1
            if counter % 10 == 0:
                await streamedMessage.edit(content=full_reply_content)
                #print(full_reply_content)
    await streamedMessage.edit(content=fullMessage)
  
    #full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    #print(full_reply_content)
    history.append({"role": "assistant", "content": fullMessage})
    return fullMessage


def get_first_500_words(url, numWords):
    
    # Set up logging mechanism
    logging.basicConfig(filename='scraping.log', level=logging.ERROR)

    try:
        # Set User-Agent to avoid getting blocked by some websites
        headers = {'User-Agent': 'Mozilla/5.0'}
        # Set a timeout to avoid getting stuck on slow sites
        response = requests.get(url, headers=headers, timeout=10)
        # Specify the encoding to avoid decoding issues
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        words = text.split()
        first_500_words = words[:numWords]
        return ' '.join(first_500_words)
    except requests.exceptions.RequestException as e:
        # Log the error and include the URL that caused the error
        logging.error(f"Error while scraping URL {url}: {str(e)}")
        return (f"Sorry, scraping {url} errored out.")

#summarize a single url
def summarize(url):
    scrapedSummaryUrl = get_first_500_words(url,2000)
    try:
        summarizedArticle = ask_openai(f"Please summarize the following information into bullet points. Highlight the most important information at the end. Article: {scrapedSummaryUrl}",history)
    except Exception as e:
        print(e)
        return('Shoot..Something went wrong or timed out.')
    return summarizedArticle

#googling function
def deepGoogle(query):
    global url1, url2, prompt_token_count, cleanedBotSearchGen, url3
    
    usersQuestion = query
    try:
        botSearchGen = ask_openai(f"I would like to search Google for {usersQuestion}. Please generate a useful and effective search query and reply ONLY with that updated query. Use the current date: '{fullDate}' and/or location: '{location}' ONLY if you think it will make the search better. Don't add date/location if it isn't applicable, for example local questions would benefit from location being added. don't use emoji's here. Remember, answer ONLY with the query that will be sent to Google.",history)
    except Exception as e:
        print(e)
        return('Shoot..Something went wrong or timed out.')

    service = build("customsearch", "v1", developerKey=googleApiKey)
    cleanedBotSearchGen=botSearchGen.strip('"')
    print(f"Searching for {cleanedBotSearchGen}")
    result = service.cse().list(
        q=cleanedBotSearchGen,
        cx=googleEngineID
    ).execute()
    print("Processing URLs...")
    try:
        url1 = result['items'][0]['link']
        url2 = result['items'][1]['link']
        url3 = result['items'][2]['link']        
    except (TypeError, KeyError):
        print("No URLs found, try rewording your search.")
        #raise ValueError("No URLs found, try rewording your search.")
    
    print("Scraping...")
    #scrape these results with beautiful soup.. mmmm
    scraped1 = get_first_500_words(url1,500)
    scraped2 = get_first_500_words(url2,500)
    scraped3 = get_first_500_words(url3,500)
    #put them all in one variable
    allScraped = (scraped1 or "") + " " + (scraped2 or "") + " " + (scraped3 or "")

    #prepare results for bot
    user_search_obj = {"role": "user", "content": allScraped}
    #we save it to a variable and feed it back to the bot to give us the search results in a more natural manner
    history.append(user_search_obj)
    #clear var for next time
    allScraped = ""
       
    #print(searchReply)
    print(f"{url1} \n{url2} \n{url3}") 
    #print(searchReply)
    try:
        botReply = ask_openai(f"You just performed a Google Search and possibly have some background on the topic of my question.  Answer my question based on that background if possible. If the answer isn't in the search results, try to field it yourself but mention the search was unproductive. DO use emojis. My question: {query}",history)
        return(botReply)
    except Exception as e:
        print(e)
        return("Shoot..sorry. I found the following urls but can't comment on them at the moment.")

def imgGen(imgPrompt):
    response = openai.Image.create(
    prompt=imgPrompt, n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return(image_url)

def resetConvoHistory():
    global history, totalCost, totalTokens, identity, imgGenNum
    history = []
    setDate()
    print(f"History reset to: {str(history)}")
    totalCost = 0
    totalTokens = 0
    imgGenNum = 0
    return
    
#---DISCORD SECTION---#
# create a Discord client object with the necessary intents
intents = discord.Intents.all()
intents.members = True
client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix='/', intents=intents)

@client.event
async def on_ready():
    reminder_channel_id = '1090120937472540903'
    print('Logged in as {0.user}'.format(client))
    print('Setting Date...')
    setDate()
"""    
#defs to remind me of things
    async def remind_exercises():
        while True:
            now = datetime.now()  # Get the current datetime
            if now.hour == 17 and now.minute == 00:
                channel = await client.fetch_channel(reminder_channel_id)
                botmessage = await channel.send("Make sure to do your physio.\n- wall stretch - 20 total\n- chair pushups - 20 total\n- 15 rows with 10 tricep extensions per arm\n- 20 shrugs\n- corner stretch")
                await asyncio.sleep(14400) 
                await botmessage.delete()
            await asyncio.sleep(60)  # Wait for 1 min before checking again
    #start timer loop
    client.loop.create_task(remind_exercises())

    async def daily_weather():
        while True:
            now = datetime.now()  # Get the current datetime
            if now.hour == 7 and now.minute == 45:
                channel = await client.fetch_channel(reminder_channel_id)
                positiveMessage = ask_openai("It's the morning, please provide me with a positive message to start my day with.",history)                
                botmessage1 = await channel.send(positiveMessage)
                resetConvoHistory()
                searchReply = deepGoogle("What is the weather forecast for Peterborough Ontario Canada today?")
                botmessage2 = await channel.send(searchReply)
                resetConvoHistory()
                await asyncio.sleep(14500) 
                await botmessage1.delete()
                await asyncio.sleep(60)
                await botmessage2.delete()    
            await asyncio.sleep(60)  # Wait for 1 min before checking again
    #start timer loop
    client.loop.create_task(daily_weather())
"""
@client.event
async def on_message(message):
    global identity
    global history
    global totalCost
    global totalTokens
    global pictureTokens
    global imgGenNum
    global outputFile

    #set name (and soon to be picture) to Wheatley by default
    """member=message.guild.me
    await member.edit(nick='Wheatley')
    with open('/home/stavros/DiscordGPT/wheatley/wheatleyComic.png', 'rb') as image:
            await client.user.edit(avatar=image.read())"""
    userName = message.author
    mention = userName.mention
    userMessage = message.content

    #message functions to easily print in color boxes
    async def blueMessage(messageToSend):
        discembed = discord.Embed(
            description=f"{messageToSend}",
            color=discord.colour.Colour.dark_blue()
        )
        bot_message = await message.channel.send(embed=discembed)
        return bot_message
    async def yellowMessage(messageToSend):
        discembed = discord.Embed(
            description=f"{messageToSend}",
            color=discord.colour.Colour.yellow()
        )
        bot_message = await message.channel.send(embed=discembed)
        return bot_message
    async def goldMessage(messageToSend):
        discembed = discord.Embed(
            description=f"{messageToSend}",
            color=discord.colour.Colour.dark_gold()
        )
        bot_message = await message.channel.send(embed=discembed)
        return bot_message
    async def redMessage(messageToSend):
        discembed = discord.Embed(
            description=f"{messageToSend}",
            color=discord.colour.Colour.red()
        )
        await message.channel.send(embed=discembed)
        return bot_message
    async def greenMessage(messageToSend):
        discembed = discord.Embed(
            description=f"{messageToSend}",
            color=discord.colour.Colour.dark_green()
        )
        bot_message = await message.channel.send(embed=discembed)
        return bot_message
    
    # ignore messages sent by the bot itself to avoid infinite loops
    if message.author == client.user:
        return
    elif '!reset' in message.content or '!thanks' in message.content or '!forget' in message.content:
        member=message.guild.me
        #await member.edit(nick='Wheatley')
        #clear chat history except for starting prompt
        resetConvoHistory()
        await blueMessage("OK. What's next?")
        calculateCost()
        await goldMessage(costing)
        return
        
    elif '!search' in message.content:
        #wipe history as this could get big
        resetConvoHistory()      
        bot_message = await message.channel.send("Searching...Please allow up to 50 seconds for a result.\n", file=discord.File('wheatley-3-blue-30sec.gif') )
        searchReply = deepGoogle(message.content[7:])
        await bot_message.delete()
        await yellowMessage(f"Searched for: {cleanedBotSearchGen}")
        await blueMessage(f"{searchReply}")
        #specifically not in boxes so as to generate thumbnails
        #await message.channel.send(f"{url1} \n{url2} \n{url3}")
        #removing thumbnails for cleaner interface
        await yellowMessage(f"{url1} \n{url2} \n{url3}")
        calculateCost()
        await goldMessage(costing)
        return
        
    elif '!summarize' in message.content:
        resetConvoHistory()
        bot_message = await message.channel.send(f"Summarizing...Please allow up to 50 seconds for a result.\n", file=discord.File('wheatley-3-blue-30sec.gif'))
        searchReply = summarize(message.content[10:])
        await bot_message.delete()
        await blueMessage(f"{searchReply}")
        calculateCost()
        await goldMessage(costing)
        return
        """    
    elif '!image' in message.content:
        bot_message = await message.channel.send(f"Generating Image...\n", file=discord.File('wheatley-3-blue-30sec.gif'))
        #bot_message = await greenMessage(f"Generating image...\n\u23F3")
        imgURL = imgGen(message.content[7:])
        await bot_message.delete()
        #set up embedded image post
        discembed = discord.Embed()
        discembed.set_image(url=imgURL)
        await message.channel.send(embed=discembed)
        imgGenNum += 1
        calculateCost()
        await goldMessage(costing)
        return
        """
    elif '!ignore' in message.content:
        print("Ignoring input.")
        await blueMessage("I did not see that.")
        return
    elif '!snake' in message.content:
        member=message.guild.me
        await member.edit(nick='Snake')
        identity = snake
        resetConvoHistory()
        await blueMessage("\U0001F40D Snake, at your service. Ask me your Python questions, I'm ready. \U0001F40D")
        return
    elif '!glados' in message.content:
        member=message.guild.me
        await member.edit(nick='GlaDOS')
        identity = glados
        resetConvoHistory()
        await blueMessage("\U0001F916 GLaDOS, at your service. What's up?\U0001F916")
        return
    elif len(message.attachments) == 1: #if file has an attachment
        #get the attached file and read it
        inputFile = message.attachments[0]
        print("Reading File")
        inputContent = await inputFile.read()
        inputContent = inputContent.decode('utf-8')
        #so inputContent is the message to be openai-ified
        bot_message = await message.channel.send(file=discord.File('wheatley-3-blue-30sec.gif'))
        discordResponse = ask_openai(inputContent,history)
        await bot_message.delete()
        with open(outputFile, "w") as responseFile:
            responseFile.write(discordResponse)
        await message.channel.send(file=discord.File(outputFile))
        await blueMessage("Please see my response in the attached file.")
        calculateCost()
        await goldMessage(costing)
        return
   
    
    #prints to terminal only - for debugging purposes   
    print(f"{userName} just said: {userMessage}")
    #this part posts an hourglass as a question as soon as the user presses enter to send their request
    # bot_message = await blueMessage('\u23F3')

    try:
        #sends users question to openai
        await stream_openai(message.content,history,message.channel)
        calculateCost()
        await goldMessage(costing)
    except Exception as e:
        print(e)
        await redMessage('Shoot..Something went wrong or timed out.')

    """bot_message = await message.channel.send(file=discord.File('wheatley-3-blue-30sec.gif'))
    try:
        #sends users question to openai
        discordResponse = ask_openai(userMessage,history)
        #at this point the respons has come back, so then you delete the 'bot_message' (the hourglass)
        await bot_message.delete()
        #debug of the response to terminal
        print(f"Bot just said: {discordResponse}")
        # send the response back to Discord
        await blueMessage(discordResponse)
        calculateCost()
        await goldMessage(costing)
    except Exception as e:
        print(e)
        await bot_message.delete()
        await redMessage('Shoot..Something went wrong or timed out.')"""

client.run(discordBotToken)
#---/DISCORD SECTION---#
#git testing
