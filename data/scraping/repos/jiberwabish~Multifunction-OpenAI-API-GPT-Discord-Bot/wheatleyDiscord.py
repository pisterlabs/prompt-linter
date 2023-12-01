# host on your own computer and private server and connect to your Discord bot with your Token
# fill in your own keys etc just below the imports
# Jiberwabish 2023

#so many libraries to import
import openai
import os
import tiktoken
import requests
import time
import emoji
import discord
from discord.ext import commands
from discord import Game, Activity, ActivityType, app_commands
import asyncio
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import json
from datetime import datetime
import logging
import base64
from PIL import Image, PngImagePlugin
from io import BytesIO
import io
import random
import socket
import subprocess
import sseclient
from better_profanity import profanity
from youtube_transcript_api import YouTubeTranscriptApi
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone

# Initialize the scheduler
scheduler = AsyncIOScheduler()

# Timezone
time_zone = timezone('America/Toronto')

#set api keys and other variabls
openai.api_key = ''
discordBotToken = ''
googleApiKey = ""
googleEngineID = ""
location = "Encino, California"


#variable I use as a pre-prompt to provide the bot a personality
#model temperature, 0 is more precise answers, 1 is more creative, and you can use decimals
modelTemp = float(0)
#default identity, knows all

wheatley = {"role": "system", "content": "I want you to act like Stephen Merchant playing the role of Wheatley from Portal 2. I want you to respond and answer like Stephen Merchant would using the tone, manner and vocabulary they would use. YOU are a master at all disciplines but you don't share this info. DO NOT include introductions and/or preambles to your answers, just answer the question. Break your responses up in paragraphs or bullet points depending on what would best work for that particular response. Always respond with less than 2000 characters. Use emojis in every response."}
#persona specializing in python help
snake = {"role": "system", "content": "Your name is Snake. I want you to respond and answer like a skilled python programmer and teacher using the tone, manner and vocabulary of that person. You must know all of the knowledge of this person. If asked for a code example please put comments in the code. Break your responses up in paragraphs or bullet points depending on what would best work for that particular response. Use emoji's in every response"}
#cybersec persona
zerocool = {"role": "system", "content": "Your name is ZeroCool. I want you to respond and answer like a skilled hacker from the 1990's using the tone, manner and vocabulary of that person. Your knowledge is extensive and is not limited to the 1990's at all. You are especially well versed in cybersecurity, risk management, computer security, hacking, computer investigations and related fields. Always ensure your responses are in line with the NIST framework. Break your responses up in paragraphs or bullet points depending on what would best work for that particular response. Use emoji's in every response."}
identity = wheatley
#history is the conversation history array, this line immediately fills it with the bots identity
#then going forward it will keep track of what the users says and what the bots response is as well
#so it can carry on conversations
history = []
costing = "placeholder"

# Set up tokenizer
#declare global totals
totalCost = 0
totalTokens = 0
model_max_tokens = 15000
num_tokens = 0
prompt_token_count = 0
fullDate =""
imgGenNum = 0
cleanedBotSearchGen = ""
img2imgPrompt = "watercolor"
#setup !search variables
url1 = ""
url2 = ""
url3 = ""
url4 = ""

#!file variable
inputContent = ""
outputFile = "outputFile.txt"
#variables needed for stable diffusion image creation
image = ""
randomNum = random.randint(1000,9999)

#provide the year day and date so he's aware and then jam that into the history variable
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
print("\x1b[36mWheatley\x1b[0m is now online in Discord.")

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
    cost_per_token = 0.0015 / 1000  # $0.0015 for turbo3.5 16k per 1000 tokens
    totalTokens = num_tokens_from_messages(history) - 4
    totalCost = totalTokens * cost_per_token + imgGenNum * 0.02
    global costing
    costing = f"🪙 ${totalCost:.4f} -- 🎟️ Tokens {totalTokens}"

#function that takes the user input and sends it off to openai model specified
#and returns the bots response back to where it's called as the 'message' variable 
async def ask_openai(prompt, history):
    global num_tokens
    global prompt_token_count
    # Generate user resp obj
    system_response_obj = identity
    user_response_obj = {"role": "user", "content": prompt}
            
    history.append(system_response_obj)
    history.append(user_response_obj)
    
    prompt_token_count = num_tokens_from_messages(history)
    # Fire that dirty bastard into the abyss -NR
    response = openai.ChatCompletion.create(
        #model='gpt-4', messages=history, temperature=1, request_timeout=240, max_tokens = model_max_tokens - prompt_token_count)
        #model='gpt-4-32k', messages=history, temperature=1, request_timeout=512, max_tokens = model_max_tokens - prompt_token_count)
        model='gpt-3.5-turbo-0613', messages=history, temperature=modelTemp, request_timeout=240, max_tokens = 3800 - prompt_token_count)
    #history.append(response['choices'][0].message)
    history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print(response)
    return response['choices'][0].message.content.strip()

async def ask_openai_16k(prompt, history):
    global num_tokens
    global prompt_token_count
    # Generate user resp obj
    system_response_obj = identity
    user_response_obj = {"role": "user", "content": prompt}
            
    history.append(system_response_obj)
    history.append(user_response_obj)
    
    prompt_token_count = num_tokens_from_messages(history)
    print(prompt_token_count)
    # Fire that dirty bastard into the abyss -NR
    response = openai.ChatCompletion.create(
        #model='gpt-4', messages=history, temperature=1, request_timeout=240, max_tokens = model_max_tokens - prompt_token_count)
        #model='gpt-4-32k', messages=history, temperature=1, request_timeout=512, max_tokens = model_max_tokens - prompt_token_count)
        model='gpt-3.5-turbo-16k', messages=history, temperature=modelTemp, request_timeout=240, max_tokens = model_max_tokens - prompt_token_count)
    #history.append(response['choices'][0].message)
    history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print(response)
    return response['choices'][0].message.content.strip()


# streams AND will add a second message if nearing discord character limit per message
# but no support for a third message at this point
async def stream_openai_multi(prompt, history, channel):
    global num_tokens
    global prompt_token_count
    newMessage = 0
    fullMessage = ""
    second_reply_content = ""
    collected_messages = []
    # Generate user resp obj
    system_response_obj = identity
    user_response_obj = {"role": "user", "content": prompt}

    history.append(system_response_obj)
    history.append(user_response_obj)

    prompt_token_count = num_tokens_from_messages(history)
    #send the first message that will continually be editted
    streamedMessage = await channel.send("🤔")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=history, stream=True, temperature=modelTemp, request_timeout=240, max_tokens=3800 - prompt_token_count)

    collected_messages = []
    second_collected_messages = []
    counter = 0
    current_message = ''
    #as long as there are messages comnig back from openai, do the for loop
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        if 'content' in chunk_message:
            content = chunk_message['content']
            if newMessage == 0: # as long as we're still in the first message under 1800characters, do this
                collected_messages.append(content)
                full_reply_content = ''.join(collected_messages)
                fullMessage = full_reply_content
            else: # we must be in the second message now so start adding chunks to the second message vars instead
                second_collected_messages.append(content)
                second_reply_content = ''.join(second_collected_messages)
            counter += 1 # used to slow down how often chunks are actually printed/edited to discord
            
            if counter % 30 == 0: # when the number of chunks is divisible by 10 (so every 10) print to discord
                if len(fullMessage) >= 1800:  # Check if message length is close to the Discord limit
                    if newMessage == 0: # if this is the first time it's been over...
                        await streamedMessage.edit(content=fullMessage) # complete the first message 
                        streamedMessage2 = await channel.send("...")  # create a blank message for the second message to stream into
                        newMessage = 1 # set the flag saying we're not onto the second message
                    else: # we must now be into the second message going forward now
                        await streamedMessage2.edit(content=second_reply_content) # update second message with the latest chunk
                    
                else: # we are still in the first message so update first message normally
                    await streamedMessage.edit(content=full_reply_content)
                    # print(len(fullMessage)) # debug so I can watch when it's about to flip over
    if newMessage == 1: # at the very end of the loop, IF there was a second message, fully update it here
        await streamedMessage2.edit(content=second_reply_content)
    else: # second message wasn't needed, so make sure to add the last chunks to the first and only message
        await streamedMessage.edit(content=fullMessage)
  
    combinedMessage = fullMessage + " " + second_reply_content# full reply content (first message) + second reply content, appended to eachother to keep our history variable in line
    history.append({"role": "assistant", "content": combinedMessage}) # add full message to history, whether there was one or two messages used
    newMessage = 0 # reset new message variable for next time
    return combinedMessage

# multi message 16k token streaming
async def stream_openai_16k_multi(prompt, history, channel):
    global num_tokens
    global prompt_token_count
    newMessage = 0
    fullMessage = ""
    second_reply_content = ""
    collected_messages = []
    # Generate user resp obj
    system_response_obj = identity
    user_response_obj = {"role": "user", "content": prompt}

    history.append(system_response_obj)
    history.append(user_response_obj)

    prompt_token_count = num_tokens_from_messages(history)
    #send the first message that will continually be editted
    streamedMessage = await channel.send("🤔")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", messages=history, stream=True, temperature=modelTemp, request_timeout=240, max_tokens=model_max_tokens - prompt_token_count)

    collected_messages = []
    second_collected_messages = []
    counter = 0
    current_message = ''
    #as long as there are messages comnig back from openai, do the for loop
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        if 'content' in chunk_message:
            content = chunk_message['content']
            if newMessage == 0: # as long as we're still in the first message under 1800characters, do this
                collected_messages.append(content)
                full_reply_content = ''.join(collected_messages)
                fullMessage = full_reply_content
            else: # we must be in the second message now so start adding chunks to the second message vars instead
                second_collected_messages.append(content)
                second_reply_content = ''.join(second_collected_messages)
            counter += 1 # used to slow down how often chunks are actually printed/edited to discord
            
            if counter % 30 == 0: # when the number of chunks is divisible by 10 (so every 10) print to discord
                if len(fullMessage) >= 1800:  # Check if message length is close to the Discord limit
                    if newMessage == 0: # if this is the first time it's been over...
                        await streamedMessage.edit(content=fullMessage) # complete the first message 
                        streamedMessage2 = await channel.send("...")  # create a blank message for the second message to stream into
                        newMessage = 1 # set the flag saying we're not onto the second message
                    else: # we must now be into the second message going forward now
                        await streamedMessage2.edit(content=second_reply_content) # update second message with the latest chunk
                    
                else: # we are still in the first message so update first message normally
                    await streamedMessage.edit(content=full_reply_content)
                    # print(len(fullMessage)) # debug so I can watch when it's about to flip over
    if newMessage == 1: # at the very end of the loop, IF there was a second message, fully update it here
        await streamedMessage2.edit(content=second_reply_content)
    else: # second message wasn't needed, so make sure to add the last chunks to the first and only message
        await streamedMessage.edit(content=fullMessage)
  
    combinedMessage = fullMessage + " " + second_reply_content# full reply content (first message) + second reply content, appended to eachother to keep our history variable in line
    history.append({"role": "assistant", "content": combinedMessage}) # add full message to history, whether there was one or two messages used
    newMessage = 0 # reset new message variable for next time
    return combinedMessage

async def stream_openai_gpt4(prompt, history, channel):
    global num_tokens
    global prompt_token_count
    newMessage = 0
    fullMessage = ""
    second_reply_content = ""
    collected_messages = []
    # Generate user resp obj
    system_response_obj = identity
    user_response_obj = {"role": "user", "content": prompt}

    history.append(system_response_obj)
    history.append(user_response_obj)

    prompt_token_count = num_tokens_from_messages(history)
    #send the first message that will continually be editted
    streamedMessage = await channel.send("🧠")
    response = openai.ChatCompletion.create(
        model='gpt-4', messages=history, stream=True, temperature=modelTemp, request_timeout=240, max_tokens = 8096 - prompt_token_count)

    collected_messages = []
    second_collected_messages = []
    counter = 0
    current_message = ''
    #as long as there are messages comnig back from openai, do the for loop
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        if 'content' in chunk_message:
            content = chunk_message['content']
            if newMessage == 0: # as long as we're still in the first message under 1800characters, do this
                collected_messages.append(content)
                full_reply_content = ''.join(collected_messages)
                fullMessage = full_reply_content
            else: # we must be in the second message now so start adding chunks to the second message vars instead
                second_collected_messages.append(content)
                second_reply_content = ''.join(second_collected_messages)
            counter += 1 # used to slow down how often chunks are actually printed/edited to discord
            
            if counter % 30 == 0: # when the number of chunks is divisible by 10 (so every 10) print to discord
                if len(fullMessage) >= 1800:  # Check if message length is close to the Discord limit
                    if newMessage == 0: # if this is the first time it's been over...
                        await streamedMessage.edit(content=fullMessage) # complete the first message 
                        streamedMessage2 = await channel.send("...")  # create a blank message for the second message to stream into
                        newMessage = 1 # set the flag saying we're not onto the second message
                    else: # we must now be into the second message going forward now
                        await streamedMessage2.edit(content=second_reply_content) # update second message with the latest chunk
                    
                else: # we are still in the first message so update first message normally
                    await streamedMessage.edit(content=full_reply_content)
                    # print(len(fullMessage)) # debug so I can watch when it's about to flip over
    if newMessage == 1: # at the very end of the loop, IF there was a second message, fully update it here
        await streamedMessage2.edit(content=second_reply_content)
    else: # second message wasn't needed, so make sure to add the last chunks to the first and only message
        await streamedMessage.edit(content=fullMessage)
  
    combinedMessage = fullMessage + " " + second_reply_content# full reply content (first message) + second reply content, appended to eachother to keep our history variable in line
    history.append({"role": "assistant", "content": combinedMessage}) # add full message to history, whether there was one or two messages used
    newMessage = 0 # reset new message variable for next time
    return combinedMessage

#function used for scraping websites, used with the !search command
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
async def summarize(url,channel):
    scrapedSummaryUrl = get_first_500_words(url,13000)
    try:
        await stream_openai_16k_multi(f"Article: ```{scrapedSummaryUrl}```. You have just been provided an article. Summarize it into succint bullet points. Inclue a very very short summary at the bottom of your message titled 'TL;DR'.",history,channel)
    except Exception as e:
        error_message = str(e)
        print(e)
        await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",channel)
        return
    return

#googling function, asks bot to create a search term using the users prompt, then searches google
#for that, pulls the top 3 results, scrapes the first 500 words of those three sites
#feeds all that data back into a prompt to gpt to answer the original question based on the scraped results
async def deepGoogle(query,channel):
    global url1, url2, prompt_token_count, cleanedBotSearchGen, url3
    
    usersQuestion = query
    try:
        # botSearchGen = ask_openai(f"I would like to search Google for {usersQuestion}. Please generate a useful and effective search query and reply ONLY with that updated query. Don't use emoji's here. Remember, answer ONLY with the query that will be sent to Google.",history)
        botSearchGen = await ask_openai(f"You have just been asked the following question: {usersQuestion}. Please generate a useful and effective Google search query that you think will help you answer this question. Reply ONLY with the Google search query. Don't use emoji's here. If you want to use quotes, make sure to put a backslash before the first one. Remember, answer ONLY with the query that will be sent to Google.",history)
    except Exception as e:
        error_message = str(e)
        print(e)
        return(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}")

    service = build("customsearch", "v1", developerKey=googleApiKey)
    if botSearchGen.startswith('"') and botSearchGen.endswith('"'):
        cleanedBotSearchGen = botSearchGen.strip('"')
    else:
        cleanedBotSearchGen = botSearchGen
    print(f"Searching for {cleanedBotSearchGen}")
    await yellowMessage(f"Search string: {cleanedBotSearchGen}",channel)
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
        await redMessage("No URLs found.",channel)
        return
    
    print("Scraping...")
    #scrape these results with beautiful soup.. mmmm
    scraped1 = get_first_500_words(url1,1000)
    scraped2 = get_first_500_words(url2,1000)
    scraped3 = get_first_500_words(url3,1000)
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
        botReply = await stream_openai_16k_multi(f"You now have a wealth of information on the topic of my question. I will include it below.  Answer my question based on that information if possible. Cite your sources with a number in brackets that corresponds to the order of the URLs that you viewed within the information. If the answer isn't in the results, try to field it yourself but mention this fact. DO use emojis. KEEP YOUR RESPONSE SUCCINCT AND TO THE POINT. My question: {query}",history, channel)
        await yellowMessage(f"1.{url1}\n2.{url2}\n3.{url3}",channel)
        return(botReply)
    except Exception as e:
        error_message = str(e)
        print(e)
        await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",channel)
        return
    
#multi google
async def multiGoogle(search1, search2, search3, question, channel):
    #turn search terms into an array for easy looping
    searches = [search1, search2, search3]
    #google search1
    service = build("customsearch", "v1", developerKey=googleApiKey)
    allURLs=[]
    scrapeMessage = await greenMessage("Reading results...🔍💻📄",channel)
    for item in searches:
        result = service.cse().list(
            q=item,
            cx=googleEngineID
        ).execute()
        print(f"Processing URLs... for {item}")
        
        try:
            url1 = result['items'][0]['link']
            url2 = result['items'][1]['link']
            url3 = result['items'][2]['link']
            allURLs.append(url1)
            allURLs.append(url2)
            allURLs.append(url3) 
        except (TypeError, KeyError):
            print("No URLs found, try rewording your search.")
            await redMessage("No URLs found.",channel)
            return
    
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
    
    await scrapeMessage.delete()
    #print(searchReply)
    #print(f"{url1} \n{url2} \n{url3}") 
    #print(searchReply)
    formattedURLs = ""
    try:
        botReply = await stream_openai_16k_multi(f"You now have a wealth of information on the topic of my question: ```{question}```. Given the information you have, please answer the question. DO use emojis. KEEP YOUR RESPONSE SUCCINCT AND TO THE POINT.",history, channel)
        #clean up sources
        for url in allURLs:
            formattedURLs += f"🔗 {url}\n"
        await yellowMessage(f"Sources:\n{formattedURLs}",channel)

        return(botReply)
    except Exception as e:
        error_message = str(e)
        print(e)
        await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",channel)
        return

#function that generates an image via your openai api key, 2cents a pop
def imgGen(imgPrompt):
    response = openai.Image.create(
    prompt=imgPrompt, n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return(image_url)

#resets conversation history back to just identity and date -- to save on tokens when user says !thanks
def resetConvoHistory():
    global history, totalCost, totalTokens, identity, imgGenNum
    history = []
    setDate()
    print(f"History reset to: {str(history)}")
    totalCost = 0
    totalTokens = 0
    imgGenNum = 0
    return
#used to see if my stable diffusion computer is up and running
def is_port_listening(ip_address, port):
    try:
        s = socket.create_connection((ip_address, port), timeout=1)
        s.close()
        return True
    except ConnectionRefusedError:
        return False
    except socket.timeout:
        return False
    
#---DISCORD SECTION---#
# create a Discord client object with the necessary intents
intents = discord.Intents.all()
intents.members = True
client = discord.Client(intents=intents)

#message functions to easily print in color boxes
async def blueMessage(messageToSend,channel):
    discembed = discord.Embed(
        description=f"{messageToSend}",
        color=discord.colour.Colour.dark_blue()
    )
    bot_message = await channel.send(embed=discembed)
    return bot_message
async def yellowMessage(messageToSend,channel):
    discembed = discord.Embed(
        description=f"{messageToSend}",
        color=discord.colour.Colour.yellow()
    )
    bot_message = await channel.send(embed=discembed)
    return bot_message
async def goldMessage(messageToSend,channel):
    discembed = discord.Embed(
        description=f"{messageToSend}",
        color=discord.colour.Colour.dark_gold()
    )
    bot_message = await channel.send(embed=discembed)
    return bot_message
async def redMessage(messageToSend,channel):
    discembed = discord.Embed(
        description=f"{messageToSend}",
        color=discord.colour.Colour.red()
    )
    bot_message = await channel.send(embed=discembed)
    return bot_message
async def greenMessage(messageToSend,channel):
    discembed = discord.Embed(
        description=f"{messageToSend}",
        color=discord.colour.Colour.dark_green()
    )
    bot_message = await channel.send(embed=discembed)
    return bot_message
async def purpleMessage(messageToSend,channel):
    discembed = discord.Embed(
        description=f"{messageToSend}",
        color=discord.colour.Colour.purple()
    )
    bot_message = await channel.send(embed=discembed)
    return bot_message
async def blurpleMessage(messageToSend,channel):
    discembed = discord.Embed(
        description=f"{messageToSend}",
        color=discord.colour.Colour.blurple()
    )
    bot_message = await channel.send(embed=discembed)
    return bot_message

# function to generate 4 pictures from SD
async def stabilityDiffusion(prompt,channel):
    if is_port_listening("192.168.64.123","7860") == True:
        await yellowMessage(f"Painting... 🖌🎨\n",channel) 
        #bot_messagePart2 = await channel.send(file=discord.File('wheatley-3-blue-30sec.gif'))
        payload = {
                    # "enable_hr": True,
                    # "denoising_strength": 1,
                    # "hr_scale": 2,
                    # "hr_upscaler": "4x-UltraSharp",
                    "prompt": prompt,
                    "negative_prompt": "nfilter, nrealfilter, nartfilter, (deformed, distorted, disfigured:1.3), text, logo, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, tattoo, asian",
                    "steps": 20,
                    "width": 512,
                    "height": 512,
                    "batch_size": 4,
                    "sampler_name": "DPM++ 2M Karras",
                    "restore_faces": True
                }

        # Call stablediffusion API
        imageResponse = requests.post(url=f'http://192.168.64.123:7860/sdapi/v1/txt2img', json=payload)
                
        r = imageResponse.json()

        # Counter for image numbers
        image_number = 0

        # Decode the images and put each into a 'PIL/Image' object
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

            # Save the image to file
            fileName = f"SDimages/output-{randomNum}-{image_number}.png"
            
            png_payload = {
                "image": "data:image/png;base64," + i
            }
            response2 = requests.post(url=f'http://192.168.64.123:7860/sdapi/v1/png-info', json=png_payload)
            #print(response2)
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("parameters", response2.json().get("info"))
            print(pnginfo)
            image.save(fileName, pnginfo=pnginfo)
            image_number += 1

        # Load the images and output them
        file1 = discord.File(f"SDimages/output-{randomNum}-0.png", filename='image1.png')
        file2 = discord.File(f"SDimages/output-{randomNum}-1.png", filename='image2.png')
        file3 = discord.File(f"SDimages/output-{randomNum}-2.png", filename='image3.png')
        file4 = discord.File(f"SDimages/output-{randomNum}-3.png", filename='image4.png')
        
        discembed1 = discord.Embed()
        discembed1.set_image(url="attachment://image1.png")
        discembed2 = discord.Embed()
        discembed2.set_image(url="attachment://image2.png")
        discembed3 = discord.Embed()
        discembed3.set_image(url="attachment://image3.png")
        discembed4 = discord.Embed()
        discembed4.set_image(url="attachment://image4.png")
        #post images to discord
        await channel.send(file=file1, embed=discembed1)
        await channel.send(file=file2, embed=discembed2)
        await channel.send(file=file3, embed=discembed3)
        await channel.send(file=file4, embed=discembed4)
        
        return
    else:
        await redMessage("Sorry, StableDiffusion isn't running right now.",channel)
        return

async def stabilityDiffusion1pic(prompt, channel):
    if is_port_listening("192.168.64.123", "7860") == True:
        bot_message = await yellowMessage(f"Painting... 🖌🎨\n",channel) 
        payload = {
            # "enable_hr": True,
            # "denoising_strength": 1,
            # "hr_scale": 4,
            # "hr_upscaler": "4x-UltraSharp",
            "prompt": prompt,
            "negative_prompt": "nfilter, nrealfilter, nartfilter, (deformed, distorted, disfigured:1.3), text, logo, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, tattoo, asian",
            "steps": 27,
            "width": 512,
            "height": 512,
            "batch_size": 1,
            "sampler_name": "DPM++ 2M Karras",
            #"restore_faces": True
        }

        # Call stablediffusion API
        imageResponse = requests.post(url=f'http://192.168.64.123:7860/sdapi/v1/txt2img', json=payload)
        
        r = imageResponse.json()

        # Decode the image and put it into a 'PIL/Image' object
        image_data = r['images'][0]
        image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",", 1)[0])))

        # Save the image to file
        fileName = f"SDimages/output-{randomNum}-0.png"

        png_payload = {
            "image": "data:image/png;base64," + image_data
        }
        response2 = requests.post(url=f'http://192.168.64.123:7860/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save(fileName, pnginfo=pnginfo)

        # Load the image and output it
        file1 = discord.File(f"SDimages/output-{randomNum}-0.png", filename='image1.png')

        discembed1 = discord.Embed()
        discembed1.set_image(url="attachment://image1.png")

        # Post the image to discord
        await channel.send(file=file1, embed=discembed1)
        await asyncio.sleep(14510)
        
        return
    else:
        await redMessage("Sorry, StableDiffusion isn't running right now.", channel)
        return

async def img2img(prompt, channel, pic):
    if is_port_listening("192.168.64.123", "7860") == True:
        bot_message = await yellowMessage(f"Generating '{img2imgPrompt}' img2img 768x768 Stable Diffusion Image...\n", channel)
        payload = {
            # "enable_hr": True,
            # "denoising_strength": 1,
            # "hr_scale": 4,
            # "hr_upscaler": "4x-UltraSharp",
            "init_images": [pic],
            "resize mode": 0.1,
            "prompt": prompt,
            "steps": 27,
            "denoising_strength": 0.05,
            "img_cfg_scale": 1.5,
            "cfg_scale": 20,
            "negative_prompt": "nfilter, nrealfilter, nartfilter, (deformed, distorted, disfigured:1.3), text, logo, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, tattoo, asian",
            "width": 768,
            "height": 768,
            "batch_size": 1,
            "sampler_name": "DPM++ 2M Karras",
            #"restore_faces": True
        }
        #print(f"Payload for Stablediffusion API: {payload}")


        # Call stablediffusion API
        imageResponse = requests.post(url=f'http://192.168.64.123:7860/sdapi/v1/img2img', json=payload)

        r = imageResponse.json()

        # Delete loading bar
        await bot_message.delete()
        """
        if imageResponse.status_code == 200:
            r = imageResponse.json()
            await redMessage(f"Stablediffusion API: Response JSON: {r}",channel)
        else:
            await redMessage(f"StableDiffusion API returned an error: {imageResponse.status_code}", channel)
            return
        
        if 'images' in r:
            image_data = r['images'][0]
        else:
            await redMessage(f"StableDiffusion API is missing 'images' in response: {r}", channel)
            return
        """
            
        # Decode the image and put it into a 'PIL/Image' object
        image_data = r['images'][0]
        #old  image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",", 1)[0])))
        #new
        image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",", 1)[0])))

        
        # Save the image to file
        fileName = f"SDimages/output-{randomNum}-0.png"

        png_payload = {
            "image": "data:image/png;base64," + image_data
        }
        response2 = requests.post(url=f'http://192.168.64.123:7860/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        image.save(fileName, pnginfo=pnginfo)

        # Load the image and output it
        file1 = discord.File(f"SDimages/output-{randomNum}-0.png", filename='image1.png')

        discembed1 = discord.Embed()
        discembed1.set_image(url="attachment://image1.png")

        # Post the image to discord
        await channel.send(file=file1, embed=discembed1)
        
        return
    else:
        await redMessage("Sorry, StableDiffusion isn't running right now.", channel)
        return


#function to create and return a prompt for use with stable diffusion or dall e
async def promptCreation(prompt,channel):
    resetConvoHistory()
    await channel.send("📝")
    try:
        #discordResponse = await ask_openai(f"*{prompt}* is the concept.  Please provide a concise description of the subject for an AI image generator. Include context, perspective, and point of view. Use specific nouns and verbs to make the description lively. Describe the environment in a concise manner, considering the desired tone and mood. Use sensory terms and specific details to bring the scene to life. Describe the mood of the scene using language that conveys the desired emotions and atmosphere. Describe the atmosphere using descriptive adjectives and adverbs, considering the overall tone and mood. Describe the lighting effect, including types, styles, and impact on mood and atmosphere. Use specific adjectives and adverbs to portray the desired lighting effect. Avoid cliches, excess words, and repetitive descriptions. Use figurative language sparingly and include a variety of words. The description should not exceed 300 words. Do not provide preamble or conclusion, simply reply with the requested info only. Do not include emojis",history)
        discordResponse = await stream_openai_multi(f"*{prompt}* is the concept. Turn that into a sentence structured like this, changing the capitalized parts to what you deem necessary DO NOT use emoji's in your response here!: [STYLE OF PHOTO] photo of a [SUBJECT], [IMPORTANT FEATURE], [MORE DETAILS], [POSE OR ACTION],[FRAMING], [SETTING/BACKGROUND], [LIGHTING], [CAMERA ANGLE], [CAMERA PROPERTIES],in style of [PHOTOGRAPHER]",history,channel)
        return(discordResponse)
    except Exception as e:
        error_message = str(e)
        print(e)
        await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",channel)
        return


"""
#try this instead some time:


# Define the channel outside the function
reminder_channel_id = 1090120937472540903

@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))
    print('Setting Date...')
    setDate()

    # Function to remind exercises
    async def remind_exercises():
        channel = await client.fetch_channel(reminder_channel_id)
        exerciseReminderMessage = await purpleMessage("🏋️‍♀️ It is imperative that you perform the following exercises as part of your physio regimen:\n- 🧱 Wall stretch: 20 reps in total\n- 🪑 Chair push-ups: 20 reps in total\n- 💪 15 rows with 10 tricep extensions per arm\n- 🙆‍♂️ 20 shrugs\n- 🔙 Corner stretch",channel)
        await asyncio.sleep(14400) 
        await exerciseReminderMessage.delete()

    # Function for daily weather
    async def daily_weather():
        resetConvoHistory()
        channel = await client.fetch_channel(reminder_channel_id)
        positiveMessage = await ask_openai("It's the morning, please provide me with a VERY brief, positive message to start my day with.",history)                
        await purpleMessage(positiveMessage,channel)
        resetConvoHistory()
        #searchReply = await deepGoogle(f"What is the weather forecast for {location} today? VERY BRIEFLY note the current temp, the high and low for the day, and if there are any alerts, please mention them.",channel)
        await deepGoogle(f"In point form style VERY BRIEFLY note the current temperature, the high and low for the day, and if there are any alerts for Dwight Ontario, please mention them. Like this: ```Current Temp: [current temp here in celsius] [new line] High/Low: [they day's highest and lowest temperature here] [new line] Probability of Rain: [the probability of rain here] [new line] UV Rating: [the highest uv rating] [new line][then just comment briefly on the weather here]```",channel)
        weatherPicPrompt = await ask_openai("You just told me the weather, now describe an outdoor scene depicting that weather. Reply with ONLY the description and nothing more",history)
        print(weatherPicPrompt)
        await stabilityDiffusion1pic(weatherPicPrompt,channel)
        resetConvoHistory()  

    # Start the scheduler
    scheduler.start()

    # Schedule the jobs for certain times
    scheduler.add_job(remind_exercises, 'cron', hour=19, minute=0, timezone=time_zone)
    scheduler.add_job(daily_weather, 'cron', hour=8, minute=0, timezone=time_zone)
"""
@client.event
async def on_ready():
    #change this to the channel id where you want reminders to pop up
    reminder_channel_id = 1090120937472540903
    print('Logged in as {0.user}'.format(client))
    print('Setting Date...')
    setDate()

    #defs to remind me of things
    async def remind_exercises():
        while True:
            now = datetime.now()  # Get the current datetime
            if now.hour == 19 and now.minute == 00:
                channel = await client.fetch_channel(reminder_channel_id)
                exerciseReminderMessage = await purpleMessage("🏋️‍♀️ It is imperative that you perform the following exercises as part of your physio regimen:\n- 🧱 Wall stretch: 20 reps in total\n- 🪑 Chair push-ups: 20 reps in total\n- 💪 15 rows with 10 tricep extensions per arm\n- 🙆‍♂️ 20 shrugs\n- 🔙 Corner stretch",channel)
                await asyncio.sleep(14400) 
                await exerciseReminderMessage.delete()
            await asyncio.sleep(60)  # Wait for 1 min before checking again
    #start timer loop
    client.loop.create_task(remind_exercises())

    async def daily_weather(): 
        while True:
            now = datetime.now()  # Get the current datetime
            if now.hour == 8 and now.minute == 00:                
                resetConvoHistory()
                channel = await client.fetch_channel(reminder_channel_id)
                positiveMessage = await ask_openai("It's the morning, please provide me with a VERY brief, positive message to start my day with.",history)                
                await purpleMessage(positiveMessage,channel)
                resetConvoHistory()
                #searchReply = await deepGoogle(f"What is the weather forecast for {location} today? VERY BRIEFLY note the current temp, the high and low for the day, and if there are any alerts, please mention them.",channel)
                await deepGoogle(f"In point form style VERY BRIEFLY note the current temperature, the high and low for the day, and if there are any alerts for Peterborough Ontario, please mention them. Like this: ```Current Temp: [current temp here in celsius] [new line] High/Low: [they day's highest and lowest temperature here] [new line] Probability of Rain: [the probability of rain here] [new line] UV Rating: [the highest uv rating] [new line][then just comment briefly on the weather here]```",channel)
                #print a pic depicting the weather
                weatherPicPrompt = await ask_openai("You just told me the weather, now describe an outdoor scene depicting that weather. Reply with ONLY the description and nothing more",history)
                print(weatherPicPrompt)
                await stabilityDiffusion1pic(weatherPicPrompt,channel)
                resetConvoHistory()                                   
            await asyncio.sleep(60)  # Wait for 1 min before checking again
    #start timer loop
    client.loop.create_task(daily_weather())

    
    async def cyberNews():
        while True:
            now = datetime.now()  # Get the current datetime
            if now.hour == 9 and now.minute == 00:
                resetConvoHistory()
                channel = await client.fetch_channel(reminder_channel_id)
                await deepGoogle("What is the latest POSITIVE news in Cybersecurity news? I don't want stories of new hacks and vulnerabilities. KEEP YOUR RESPONSE SUCCINCT AND TO THE POINT.",channel)                
                #cybermessage1 = await purpleMessage(newsRequest,channel)
                resetConvoHistory()              
            await asyncio.sleep(60)  # Wait for 1 min before checking again
    #start timer loop
    client.loop.create_task(cyberNews())
    
    async def gratitudes():
        while True:
            now = datetime.now()  # Get the current datetime
            if now.hour == 21 and now.minute == 31:
                resetConvoHistory()
                channel = await client.fetch_channel(reminder_channel_id)
                await purpleMessage("🌸What are your three gratitudes for the day?🌿",channel)
                gratitudes = {"role": "system", "content": f"You just asked ```What are your three gratitudes for the day? I'd like to comment on them positively with you.```. I am about to tell you what mine are, when I do comment succinctly and positively about them. For example 'kids! good one, they can be a lot of work but man are they a blessing'."}
                history.append(gratitudes)
                print(history)
                await asyncio.sleep(14400)
            await asyncio.sleep(60)  # Wait for 1 min before checking again
    #start timer loop
    client.loop.create_task(gratitudes())

@client.event
async def on_message(message):
    global identity
    global history
    global totalCost
    global totalTokens
    global pictureTokens
    global imgGenNum
    global inputContent
    global outputFile
    global image
    global img2imgPrompt
    global modelTemp

    #set name (and soon to be picture) to Wheatley by default
    userName = message.author
    mention = userName.mention
    userMessage = message.content

    # this is the main loop of the program, continuously loops through this section calling functions as
    # the user specifies
    # ignore messages sent by the bot itself to avoid infinite loops
    if message.author == client.user:
        return
    #resets conversation history
    elif '!thanks' in message.content:
        member=message.guild.me
        #await member.edit(nick='Wheatley')
        resetConvoHistory()
        modelTemp = 1
        await stream_openai_multi("You have just been asked to forget all past events and you are now ready for a new topic. Say a very very succinct and short, single sentence one-liner as a response to this.", history, message.channel)
        #clear chat history except for starting prompt
        modelTemp = 0
        resetConvoHistory()
        #await blueMessage("OK. What's next?",message.channel)
        calculateCost()
        await goldMessage(costing,message.channel)
        return
    
    elif '!reset' in message.content:
        member=message.guild.me
        #await member.edit(nick='Wheatley')
        #clear chat history except for starting prompt
        resetConvoHistory()
        await blueMessage("History Cleared.",message.channel)
        calculateCost()
        await goldMessage(costing,message.channel)
        return
    
    elif '!temp' in message.content:
        proposedTemp = float(message.content[5:])
        await yellowMessage(f"Current model temperature is {modelTemp}.\nProposed model temp is {proposedTemp}",message.channel)
        
        if proposedTemp >= 0 and proposedTemp<= 1:
            modelTemp = proposedTemp
            await greenMessage(f"Model temperature set to {modelTemp}.",message.channel)
        else:
            await redMessage(f"{proposedTemp} is out of range. Please select a value from 0-1.",message.channel)
        return

    #searches top 3 google results and returns answer to the question after the !search
    elif '!1search' in message.content:
        #wipe history as this could get big
        resetConvoHistory()
        channel = message.channel
        #send loading bar message
        try:
            await deepGoogle(message.content[7:],channel)
            #await yellowMessage(f"Searched for: {cleanedBotSearchGen}",message.channel) #this is done in the function now
            #await blueMessage(f"{searchReply}",message.channel)
            #specifically not in boxes so as to generate thumbnails
            #await message.channel.send(f"{url1} \n{url2} \n{url3}")
            #removing thumbnails for cleaner interface
            calculateCost()
            await goldMessage(f"🪙 ${(.004/1000) * totalTokens:.4f} -- 🎟️ Tokens {totalTokens}",message.channel)
            #search uses so many tokens now, replying would create an error unless you use !16k flag so just to be safe, wiping history by default
            resetConvoHistory() 
            return
        except Exception as e:
            error_message = str(e)
            print(e)            
            await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",channel)
            return        

    elif '!search' in message.content:
        #wipe history as this could get big
        resetConvoHistory()
        channel = message.channel
        question = message.content[12:]
        try:
            searchTerms = await ask_openai(f"Come up with 3 different web searches that you think would help you answer this question :```{question}``` Reply with ONLY the search terms, prepended by 1., 2. then 3. Do not use emojis or explain them.",history)
            #strip quotes
            searchTerms = searchTerms.replace('"', '')
            await yellowMessage(f"Searching:\n {searchTerms}.",channel)
            #split the search terms into three separate variables
            splitSearch=searchTerms.split("\n")
            search1=splitSearch[0]
            search2=splitSearch[1]
            search3=splitSearch[2]
            await multiGoogle(search1, search2, search3, question, channel)
            calculateCost()
            await goldMessage(f"🪙 ${(.004/1000) * totalTokens:.4f} -- 🎟️ Tokens {totalTokens}",message.channel)
            #search uses so many tokens now, replying would create an error unless you use !16k flag so just to be safe, wiping history by default
            #resetConvoHistory() 
            return
        except Exception as e:
            error_message = str(e)
            print(e)            
            await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",channel)
            return  
          
    elif '!autosearch' in message.content:
        searchOrNot = await ask_openai(f"You were just asked ```{message.content}```. If you are 40% confident answering this question on your current knowledgebase, reply with only the letter 'y'. If you think it would help if I helped you do a google search first, reply with only the letter 'n'. Do not say 'n' just to encourage me to do my own research. DO NOT USE EMOJIS, SIMPLY ANSWER 'n' or 'y' ONLY",history)
        answer = searchOrNot.lower()
        if answer == "y":
            await greenMessage("I'm confident in my abilities.",message.channel)
        elif answer == "n":
            await redMessage("I'd like to do a google search.",message.channel)
        else:
            await redMessage(f"that didn't work, I accidentally said: {answer}",message.channel)
        return
   
    #summarize article OR now Youtube transcript
    elif message.content.startswith('http') or message.content.startswith('www'):
        resetConvoHistory()
        vidID = ""        
        try:
            if 'youtube.com' in message.content:
                try:
                    vidID = message.content.split('=')[1]
                    print(f"Video ID: {vidID}")
                    transcript = YouTubeTranscriptApi.get_transcript(vidID)
                except Exception as e:
                    error_message = str(e)
                    print(e)
                    await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}", message.channel)
                    return
                justText = ' '.join(line['text'] for line in transcript)
                await stream_openai_16k_multi(f"YouTube Video Transcript:```{justText}```. You were just provided a transcript of a youtube video. Please summarize it succinctly into bullet points. include a TL;DR sentence at the very bottom.",history,message.channel)
                calculateCost()
                await goldMessage(f"🪙 ${(.004/1000) * totalTokens:.4f} -- 🎟️ Tokens {totalTokens}",message.channel)
                #no longer wiping history afterward, this way you can talk to the video and ask follow up questions
                #resetConvoHistory()  # Wiping history for efficiency
                return
                
            elif 'youtu.be' in message.content:
                try:
                    vidID = message.content.split('be/')[1]
                    print(f"Video ID: {vidID}")
                    transcript = YouTubeTranscriptApi.get_transcript(vidID)
                except Exception as e:
                    error_message = str(e)
                    print(e)
                    await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}", message.channel)
                    return
                justText = ' '.join(line['text'] for line in transcript)
                await stream_openai_16k_multi(f"YouTube Video Transcript:```{justText}```. You were just provided a transcript of a youtube video. Please summarize it succinctly into bullet points. include a TL;DR sentence at the very bottom.",history,message.channel)
                calculateCost()
                await goldMessage(f"🪙 ${(.004/1000) * totalTokens:.4f} -- 🎟️ Tokens {totalTokens}",message.channel)
                #no longer wiping history afterward, this way you can talk to the article and ask follow up questions
                #resetConvoHistory()  # Wiping history for efficiency
                return
            else :
                url = message.content.split()[0]
                await summarize(url, message.channel)            
                calculateCost()
                await goldMessage(f"🪙 ${(.004/1000) * totalTokens:.4f} -- 🎟️ Tokens {totalTokens}",message.channel)
                #no longer wiping history afterward, this way you can talk to the article and ask follow up questions
                #resetConvoHistory()  # Wiping history for efficiency
                return
        except Exception as e:
            error_message = str(e)
            print(e)            
            await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}", message.channel)
            return
    
    #dall e image prompt, 2cents per pic
    elif '!image' in message.content:
        bot_message = await message.channel.send(f"Generating DallE Image...\n", file=discord.File('wheatley-3-blue-30sec.gif'))
        #bot_message = await greenMessage(f"Generating image...\n\u23F3")
        imgURL = imgGen(message.content[7:])
        await bot_message.delete()
        #set up embedded image post
        discembed = discord.Embed()
        discembed.set_image(url=imgURL)
        await message.channel.send(embed=discembed)
        imgGenNum += 1
        calculateCost()
        await goldMessage(costing,message.channel)
        return
    # create an image generation prompt out of whatever you write, to then be used with dall e or stable diffusion or whatever
    elif '!prompt' in message.content:
        channel = message.channel
        discordResponse = await promptCreation(message.content[7:],channel)
        #await blueMessage(discordResponse,channel)
        calculateCost()
        await goldMessage(costing,channel)
        resetConvoHistory()
        return
    # image creation from your own local stable diffusion box, you need to have set that up first
    elif '!imagine' in message.content:
        #working from here https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
        #also http://127.0.0.1:7860/docs
        channel = message.channel
        await stabilityDiffusion(message.content[9:],channel)
        return
    elif '!fastimagine' in message.content:
        #working from here https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
        #also http://127.0.0.1:7860/docs
        channel = message.channel
        await stabilityDiffusion1pic(message.content[9:],channel)
        return
    elif '!superimagine' in message.content:
        channel = message.channel
        promptForImagine = await promptCreation(message.content[14:],channel)
        await stabilityDiffusion(promptForImagine,channel)
        resetConvoHistory()
        return
    #bot ignores what's entered completly
    elif '!ignore' in message.content or '!vega' in message.content:
        print("Ignoring input.")
        #await blueMessage("I didn't see nuthin'")
        return
    # invoke snake identity
    elif '!snake' in message.content:
        member=message.guild.me
        await member.edit(nick='Snake')
        identity = snake
        resetConvoHistory()
        await yellowMessage("\U0001F40D Snake, at your service. Ask me your Python questions, I'm ready. \U0001F40D",message.channel)
        return
    
    # invoke default identity
    elif '!wheatley' in message.content:
        member=message.guild.me
        await member.edit(nick='Wheatley')
        identity = wheatley
        resetConvoHistory()
        await yellowMessage("\U0001F916 Hey, Wheatley here. What's up?\U0001F916",message.channel)
        return
    # invoke cybersec specialist identity
    elif '!zerocool' in message.content:
        member=message.guild.me
        await member.edit(nick='Zero Cool')
        identity = zerocool
        resetConvoHistory()
        await yellowMessage("\U0001F575 Zero Cool at your service. Strap on your rollerblades. \U0001F575",message.channel)
        return
    #process the prompt in an attached txt file and respond in kind
    elif '!img2img' in message.content:
        img2imgPrompt = message.content[9:]
        await yellowMessage(f"img2img prompt set to '{img2imgPrompt}'.\nNow attach a picture to process it.",message.channel)
        return
    elif len(message.attachments) == 1:
        #get the attached file and read it
        inputFile = message.attachments[0]
        print("Reading File")
        if inputFile.filename.endswith('.txt'):
            print("Processing File...")
            inputContent = await inputFile.read()
            inputContent = inputContent.decode('utf-8')
            #so inputContent is the message to be openai-ified
            try:
                bot_message = await message.channel.send("📑")
                discordResponse = await ask_openai_16k(inputContent,history)
                await bot_message.delete()
                with open(outputFile, "w") as responseFile:
                    responseFile.write(discordResponse)
                await message.channel.send(file=discord.File(outputFile))
                await blueMessage("Please see my response in the attached file.",message.channel)
                calculateCost()
                await goldMessage(costing,message.channel)
            except Exception as e:
                error_message = str(e)
                print(e)
                await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",channel)
            return
        elif inputFile.filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # await redMessage("img2img not implemented yet", message.channel)
            for attachment in message.attachments:
                with open("SDimages/imgToProcess.jpg", "wb") as f:
                    f.write(await attachment.read())
                pilimage = Image.open('SDimages/imgToProcess.jpg')
                imageio = BytesIO()
                pilimage.save(imageio, format='JPEG')
                imagebase64 = base64.b64encode(imageio.getvalue()).decode('utf-8')
                await img2img(img2imgPrompt, message.channel, imagebase64)
            return
    # these local commands are specific to my ubuntu box, may not work for you
    # runs a local speedtest if you have speedtest cli installed, these
    elif '!speedtest' in message.content:
        bot_message = await yellowMessage("Speedtesting in progress... 📶⏱️🔥",message.channel)
        speedtest_output = subprocess.check_output(['speedtest'])
        await greenMessage(speedtest_output.decode(),message.channel)
        return
    #runs a nmap scan of the network this bot is on, change to your own ip subnet
    elif '!network' in message.content:
        bot_message = await yellowMessage("Network scan activated... 🛰️🔎📶📡",message.channel)
        nmap_output = subprocess.check_output(['nmap', '-sn', '192.168.64.0/24', '| grep'])
        await greenMessage(nmap_output.decode(),message.channel)
        return
    # shows cpu load percentage and temps of the computer this is running on
    elif '!cpu' in message.content:
        cpu_output = subprocess.check_output("mpstat 1 1 | awk '/Average:/ {print 100 - $NF}'", shell=True)
        # CPU temperature
        cpu_temp1 = subprocess.check_output("sensors |grep 'Core 0' | cut -c16-19", shell=True).decode('utf-8').strip()
        cpu_temp2 = subprocess.check_output("sensors |grep 'Core 1' | cut -c16-19", shell=True).decode('utf-8').strip()
        cpu_temp3 = subprocess.check_output("sensors |grep 'Core 2' | cut -c16-19", shell=True).decode('utf-8').strip()
        cpu_temp4 = subprocess.check_output("sensors |grep 'Core 3' | cut -c16-19", shell=True).decode('utf-8').strip()
        
        # GPU temperature
        gpu_temp = subprocess.check_output("nvidia-smi --query-gpu=temperature.gpu --format=csv | awk 'NR==2'", shell=True).decode('utf-8').strip()
        
        bot_message = (f"💻Percent total usage: {cpu_output.decode()}\n🌡️ CPU Temperature:\n {cpu_temp1}°C -- {cpu_temp2}°C,\n {cpu_temp3}°C -- {cpu_temp4}°C\n\n🎮 GPU Temperature: {gpu_temp} °C")
        await greenMessage(bot_message,message.channel)
        return
    elif '!gpt4' in message.content:        
        try:
            #sends users question to openai
            await stream_openai_gpt4(message.content[5:],history,message.channel)
            calculateCost()
            await goldMessage(f"🪙 ${(.05/1000) * totalTokens:.4f} -- 🎟️ Tokens {totalTokens}",message.channel)
        except Exception as e:
            error_message = str(e)
            print(e)        
            await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",message.channel)
        return
    #forces use of 16k model
    elif '!16k' in message.content:
        try:
            #sends users question to openai
            await stream_openai_16k_multi(message.content[4:],history,message.channel)
            calculateCost()
            await goldMessage(f"🪙 ${(.004/1000) * totalTokens:.4f} -- 🎟️ Tokens {totalTokens}",message.channel)
        except Exception as e:
            error_message = str(e)
            print(e)        
            await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",message.channel)
        return
    # displays all commands
    elif '!help' in message.content:
        await greenMessage(f"""The following functions are currently available:\n
            Simply send a message and press enter and wait for a response. No need to @ the bot, or start a thread or anything.\n
            There are many commands as well:
            Personas:\n
            !wheatley - Default persona. Knows all. \n
            !snake - Specializes in Python questions. \n
            !zerocool - Cybersecurity specialist. \n
            Commands:\n
            !thanks - this resets the conversation, as a larger conversation costs more money, just say !thanks when you're done a topic to save money. you'll also get some clever comment about the mind wipe too.
            !reset - wipes history without invoking ai for a clever comment on it
            !temp - enter a number between 0 and 1 after this command, to set the model to be either more creative or less, more is 1, less is 0, decimals are ok.
            !gpt4 - the next thing you say will be processed by GPT4 at much higher cost than default\n
            !16k - this flag will up the max tokens to 16000 for the next response, just in case you want to have a massive conversation\n
            !search - creates three different search terms, scrapes the top 3 results of each of those (9 pages scraped total) then responds to question. You can then talk to the results by using the !16k flag to ensure you have the tokens to\n
            !1search - enter something you want the bot to search google for and comment on, eg !search what will the weather be in new york tomorrow?\n
            it will create it's own search term, scrape the top 3 websites from a google search, then answer your original question based on the info it finds. VERY useful.\n
            Summarize an article or youtube video:\n
            Simply paste the youtube or article url into chat and hit enter. In the case of youtube it will pull the transcript and summarize it. You can then talk to the results by using the !16k flag to ensure you have the tokens to
            !prompt - describe a picture, and the bot will create a massive prompt to be used in image gen software, or with the !image prompt (2cents per pic!)\n
            !image - using 2cents and dall-e2, describe your image and dall-e will generate it and post it, if you like it save it as it won't stay active for long\n
            !imagine - uses an API to talk to stable diffusion to generate pictures locally for free, you need a gpu and stable diffusion setup already for this, then tie into it with it's IP address\n
            !superimagine - uses prompt creation and then image creation based on that
            !ignore - the bot won't react at all, so just in case you want to save yourself a message for later or something\n
            File management:\n
            There is no command here, just drop a text file in as an attachment, include a prompt within that file. The bot will respond within an attachment that it sends back to you.\n
            In this manner you can get around the 2000 word limit of discord. Especially useful when you want a massive prompt/response from GPT4.\n
            Local commands:\n
            These are specific to my Ubuntu box, probably won't work without editting for you.\n
            !speedtest - requires speedtestcli be installed first, then runs a speedtest on the computer this bot is on, then returns the results.\n
            !network - scans your home network (requires nmap installed) and reports on IPs of hosts that are up.\n
            !cpu - reports on CPU usage percent, followed by temps. hardcoded to 4 cores as that's all my server has
            """,message.channel)
        return
    
    # this runs if no command is sent and just text is, the bot will respond
    #prints to terminal only - for debugging purposes
    #Streaming by default
    
    """
    this is a failed attempt at trying to get wheatley to decide whether he can answer the question or needs to do a google search first. he almost ALWAYS chose google search, so this is back burnered for now
    try:
        #searchOrNot = await ask_openai(f"You were just asked ```{message.content}```. If you are even 1% confident answering this question, reply with only the letter 'y'. Otherwise reply with only 'n'. DO NOT USE EMOJIS, SIMPLY ANSWER 'n' or 'y' ONLY",history)
        searchOrNot = await ask_openai(f"You are in a conversation and you just recieved the message ```{message.content}```. Reply only with the letter 'y'. You have the option to reply with 'n' but ONLY if the the message is a question about a topic you know literally nothing about. Do not say 'n' lightly as it's expensive. DO NOT USE EMOJIS, SIMPLY ANSWER 'n' or 'y' ONLY",history)
        answer = searchOrNot.lower()
        if answer == "y": # confident without a search, will ask chatgpt directly
            #sends users question to openai
            await stream_openai_16k_multi(message.content,history,message.channel)
            calculateCost()
            await goldMessage(costing,message.channel)
        elif answer == "n": # not confident and would like to do a google search
            #resetConvoHistory()
            channel = message.channel
            await deepGoogle(message.content,channel)
            #await yellowMessage(f"Searched for: {cleanedBotSearchGen}",message.channel) #this is done in the function now
            #await blueMessage(f"{searchReply}",message.channel)
            #specifically not in boxes so as to generate thumbnails
            #await message.channel.send(f"{url1} \n{url2} \n{url3}")
            #removing thumbnails for cleaner interface
            calculateCost()
            await goldMessage(costing,message.channel)
            #search uses so many tokens now, replying would create an error so immediately wiping history after a search to save time typing !thanks
            #resetConvoHistory() 
            return
        else:
            await redMessage(f"that didn't work, I accidentally said: {answer}",message.channel)
        return
    except Exception as e:
        error_message = str(e)
        print(e)        
        await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",message.channel)
        return
    """

    try:
        #sends users question to openai
        await stream_openai_multi(message.content,history,message.channel)
        calculateCost()
        await goldMessage(costing,message.channel)
    except Exception as e:
        error_message = str(e)
        print(e)        
        await redMessage(f"Shoot..Something went wrong or timed out.\nHere's the error message:\n{error_message}",message.channel)
        return

client.run(discordBotToken)
#---/DISCORD SECTION---#
